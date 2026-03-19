package com.tom.rv2ide.artificial.agents.openai

import android.content.Context
import com.tom.rv2ide.artificial.services.ArtificialService
import com.tom.rv2ide.artificial.rules.WritingRules
import com.tom.rv2ide.artificial.project.awareness.ProjectTreeResult
import com.tom.rv2ide.artificial.file.AIFileWriter
import com.tom.rv2ide.artificial.file.FileWriteResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import com.tom.rv2ide.artificial.agents.Agents
import org.json.JSONObject
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL
import com.tom.rv2ide.artificial.exceptions.*
import com.tom.rv2ide.artificial.agents.AIAgent
import com.tom.rv2ide.artificial.agents.AIAgentRegistry
import com.tom.rv2ide.artificial.secrets.ApiKey
import com.tom.rv2ide.artificial.agents.ModificationAttempt
import org.slf4j.LoggerFactory

class OpenAI : AIAgent {

    private val log = LoggerFactory.getLogger(OpenAI::class.java)
    private var apiKey: String? = null
    private val writingRules = WritingRules.Instructions()
    private var projectTreeResult: ProjectTreeResult? = null
    private var fileWriter: AIFileWriter? = null
    private val conversationHistory = mutableListOf<ConversationMessage>()
    private val modificationHistory = mutableListOf<ModificationAttempt>()
    private var currentAttemptCount = 0
    private val maxRetryAttempts = 3
    private var agents: Agents? = null
    private var selectedModel: String = "gpt-4o"
    override val providerId = "openai"
    override val providerName = "OpenAI"

    companion object {
        fun registerAgent() {
            AIAgentRegistry.register("openai", object : AIAgentRegistry.AgentFactory {
                override fun create(context: Context): AIAgent {
                    return OpenAI()
                }

                override fun hasValidApiKey(): Boolean {
                    val key = ApiKey.getOpenAIApiKey()
                    val log = LoggerFactory.getLogger(OpenAI::class.java)
                    log.debug("hasValidApiKey check: {}, key length: {}", key != null && key.isNotEmpty(), key?.length ?: 0)
                    return key != null && key.isNotEmpty()
                }

                override fun getApiKey(): String? {
                    val key = ApiKey.getOpenAIApiKey()
                    val log = LoggerFactory.getLogger(OpenAI::class.java)
                    log.debug("getApiKey called, returning key of length: {}", key?.length ?: 0)
                    return key
                }
            })
        }
    }

    override fun initialize(apiKey: String, context: Context) {
        this.apiKey = apiKey
        agents = Agents(context)
        var selectedModel = agents?.getAgent() ?: "gpt-4o"
        if (selectedModel.isNullOrBlank()) {
            selectedModel = "gpt-4o"
            agents?.setAgent(selectedModel)
            agents?.setProvider("openai")
        } else {
            agents?.setProvider("openai")
        }
        this.selectedModel = selectedModel
    }

    override fun reinitializeWithNewModel(apiKey: String, context: Context) {
        initialize(apiKey, context)
    }

    override fun setContext(context: Context) {
        fileWriter = AIFileWriter(context)
    }

    override fun setProjectData(projectTreeResult: ProjectTreeResult) {
        this.projectTreeResult = projectTreeResult
    }

    override fun clearConversation() {
        conversationHistory.clear()
        modificationHistory.clear()
        currentAttemptCount = 0
    }

    override fun recordModification(filePath: String, oldContent: String?, newContent: String, success: Boolean) {
        modificationHistory.add(
            ModificationAttempt(
                timestamp = System.currentTimeMillis(),
                filePath = filePath,
                previousContent = oldContent,
                newContent = newContent,
                attemptNumber = currentAttemptCount,
                success = success
            )
        )
    }

    override fun undoLastModification(): Boolean {
        if (modificationHistory.isEmpty()) return false
        val lastMod = modificationHistory.lastOrNull { it.success } ?: return false
        if (lastMod.previousContent != null) {
            val result = writeFile(lastMod.filePath, lastMod.previousContent)
            if (result is FileWriteResult.Success) {
                modificationHistory.remove(lastMod)
                return true
            }
        } else {
            File(lastMod.filePath).delete()
            modificationHistory.remove(lastMod)
            return true
        }
        return false
    }

    override fun getModificationHistory(): List<ModificationAttempt> {
        return modificationHistory.toList()
    }

    override fun resetAttemptCount() {
        currentAttemptCount = 0
    }

    override fun incrementAttemptCount() {
        currentAttemptCount++
    }

    override fun getCurrentAttemptCount(): Int = currentAttemptCount

    override fun canRetry(): Boolean = currentAttemptCount < maxRetryAttempts

    override suspend fun generateCode(
        prompt: String,
        context: String?,
        language: String,
        projectStructure: String?,
    ): Result<String> =
        withContext(Dispatchers.IO) {
            val key = apiKey ?: return@withContext Result.failure(
                IllegalStateException("OpenAI service not initialized")
            )

            val response = callOpenAIAPI(key, prompt)

            if (response.isBlank()) {
                return@withContext Result.failure(Exception("Empty response from AI"))
            }

            conversationHistory.add(ConversationMessage("user", prompt))
            conversationHistory.add(ConversationMessage("assistant", response))

            if (conversationHistory.size > 20) {
                conversationHistory.removeAt(0)
                conversationHistory.removeAt(0)
            }

            Result.success(response)
        }

    private fun callOpenAIAPI(apiKey: String, prompt: String): String {
        if (selectedModel.startsWith("gpt-5") ||
            selectedModel.contains("codex") ||
            selectedModel.contains("o1") ||
            selectedModel.contains("o3")
        ) {
            return attemptResponses(apiKey, prompt)
        }
        return attemptChatCompletions(apiKey, prompt)
    }

    private fun attemptResponses(apiKey: String, prompt: String): String {
        val url = URL("https://api.openai.com/v1/responses")
        val connection = url.openConnection() as HttpURLConnection

        try {
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $apiKey")
            connection.doOutput = true

            val inputArray = JSONArray()

            val message = JSONObject()
            message.put("role", "user")

            val contentArray = JSONArray()
            val textObj = JSONObject()
            textObj.put("type", "text")
            textObj.put("text", writingRules.useThis() + "\n\n" + prompt)

            contentArray.put(textObj)
            message.put("content", contentArray)

            inputArray.put(message)

            val requestBody = JSONObject()
            requestBody.put("model", selectedModel)
            requestBody.put("input", inputArray)
            requestBody.put("max_output_tokens", 4096)

            connection.outputStream.use {
                it.write(requestBody.toString().toByteArray())
            }

            val responseCode = connection.responseCode

            if (responseCode != HttpURLConnection.HTTP_OK) {
                val error = connection.errorStream?.bufferedReader()?.readText()
                    ?: "Unknown error"
                throw Exception(error)
            }

            val responseBody = connection.inputStream.bufferedReader().readText()
            val json = JSONObject(responseBody)

            val output = json.optJSONArray("output")
            if (output != null && output.length() > 0) {
                val first = output.getJSONObject(0)
                val content = first.optJSONArray("content")
                if (content != null && content.length() > 0) {
                    val text = content.getJSONObject(0).optString("text")
                    if (text.isNotEmpty()) return text
                }
            }

            throw Exception("No text content returned")
        } finally {
            connection.disconnect()
        }
    }

    private fun attemptChatCompletions(apiKey: String, prompt: String): String {
        val url = URL("https://api.openai.com/v1/chat/completions")
        val connection = url.openConnection() as HttpURLConnection

        try {
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $apiKey")
            connection.doOutput = true

            val messages = JSONArray()

            val systemMessage = JSONObject()
            systemMessage.put("role", "system")
            systemMessage.put("content", writingRules.useThis())
            messages.put(systemMessage)

            val userMessage = JSONObject()
            userMessage.put("role", "user")
            userMessage.put("content", prompt)
            messages.put(userMessage)

            val requestBody = JSONObject()
            requestBody.put("model", selectedModel)
            requestBody.put("messages", messages)
            requestBody.put("max_tokens", 4096)

            connection.outputStream.use {
                it.write(requestBody.toString().toByteArray())
            }

            val responseCode = connection.responseCode

            if (responseCode != HttpURLConnection.HTTP_OK) {
                val error = connection.errorStream?.bufferedReader()?.readText()
                    ?: "Unknown error"
                throw Exception(error)
            }

            val responseBody = connection.inputStream.bufferedReader().readText()
            val jsonResponse = JSONObject(responseBody)

            val choices = jsonResponse.optJSONArray("choices")
            if (choices != null && choices.length() > 0) {
                val message = choices.getJSONObject(0)
                    .optJSONObject("message")
                val content = message?.optString("content")
                if (!content.isNullOrEmpty()) return content
            }

            throw Exception("No response content")
        } finally {
            connection.disconnect()
        }
    }

    override fun writeFile(filePath: String, content: String): FileWriteResult {
        val writer = fileWriter ?: return FileWriteResult.Error("File writer not initialized")
        return writer.writeFile(filePath, content, createBackup = true)
    }

    override fun isInitialized(): Boolean = apiKey != null
}