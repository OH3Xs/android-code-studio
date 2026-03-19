/*
 *  This file is part of AndroidCodeStudio.
 *
 *  AndroidCodeStudio is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  AndroidCodeStudio is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *   along with AndroidCodeStudio.  If not, see <https://www.gnu.org/licenses/>.
 */

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

/*
 * @author Mohammed-baqer-null @ https://github.com/Mohammed-baqer-null
 */

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
        try {
            this.apiKey = apiKey
            agents = Agents(context)
            var selectedModel = agents?.getAgent() ?: "gpt-4o"

            // Only validate that the model is not null/empty – do NOT restrict to a hardcoded list.
            if (selectedModel.isNullOrBlank()) {
                selectedModel = "gpt-4o"
                agents?.setAgent(selectedModel)
                agents?.setProvider("openai")
            } else {
                // Ensure the provider is set correctly
                agents?.setProvider("openai")
            }

            this.selectedModel = selectedModel
            log.debug("Initialized with model: {}", this.selectedModel)
        } catch (e: Exception) {
            throw e
        }
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
                modificationHistory.removeAt(modificationHistory.lastIndexOf(lastMod))
                return true
            }
        } else {
            try {
                File(lastMod.filePath).delete()
                modificationHistory.removeAt(modificationHistory.lastIndexOf(lastMod))
                return true
            } catch (e: Exception) {
                return false
            }
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

    private fun isUserRequestingCorrection(message: String): Boolean {
        val correctionKeywords = listOf(
            "wrong", "not what", "mistake", "error", "incorrect",
            "that's not", "not right", "fix", "undo", "revert",
            "different", "try again", "not working"
        )
        return correctionKeywords.any { message.lowercase().contains(it) }
    }

    override suspend fun generateCode(
        prompt: String,
        context: String?,
        language: String,
        projectStructure: String?,
    ): Result<String> =
        withContext(Dispatchers.IO) {
            try {
                val key = apiKey
                    ?: return@withContext Result.failure(
                        IllegalStateException("OpenAI service not initialized")
                    )

                // Ensure model is not null – if it is, fallback (shouldn't happen)
                if (selectedModel.isNullOrBlank()) {
                    selectedModel = "gpt-4o"
                }

                val fileContents = readRelevantFiles()
                val needsCorrection = isUserRequestingCorrection(prompt)

                val fullPrompt = buildString {
                    append("=== PROJECT STRUCTURE (THESE ARE THE EXACT PATHS YOU MUST USE) ===\n")
                    if (projectTreeResult != null) {
                        append(projectTreeResult!!.tree)
                        append("\n\n")
                        append("CRITICAL: Use ONLY the paths shown above. Do NOT make up fake paths like '/storage/emulated/0/project' or 'com.example.yourproject'.\n")
                        append("CRITICAL: Look at the actual paths above and use those EXACT paths.\n\n")
                    }

                    if (fileContents.isNotEmpty()) {
                        append("=== CURRENT FILES CONTENT ===\n")
                        fileContents.forEach { (path, content) ->
                            append("FILE: $path\n")
                            append("CONTENT:\n")
                            append(content)
                            append("\n\n")
                        }
                    }

                    if (context != null) {
                        append("=== ADDITIONAL CONTEXT ===\n")
                        append(context)
                        append("\n\n")
                    }

                    if (conversationHistory.isNotEmpty()) {
                        append("=== CONVERSATION HISTORY ===\n")
                        conversationHistory.forEach { msg ->
                            append("${msg.role.uppercase()}: ${msg.content}\n\n")
                        }
                    }

                    if (needsCorrection && modificationHistory.isNotEmpty()) {
                        append("=== CORRECTION REQUIRED ===\n")
                        append("The user indicated the previous modification was WRONG.\n")
                        append("Previous failed attempts:\n")
                        modificationHistory.takeLast(3).forEach { attempt ->
                            append("Attempt ${attempt.attemptNumber}: ${attempt.filePath}\n")
                            append("Result: ${if (attempt.success) "Applied but user rejected" else "Failed"}\n\n")
                        }
                        append("You MUST try a DIFFERENT approach. Do NOT repeat the same solution.\n")
                        append("Analyze what went wrong and provide a better solution.\n\n")
                    }

                    if (currentAttemptCount > 0) {
                        append("=== RETRY ATTEMPT $currentAttemptCount/$maxRetryAttempts ===\n")
                        append("This is retry attempt number $currentAttemptCount.\n")
                        append("Previous attempts did not satisfy the user.\n")
                        append("Think carefully and provide a different solution.\n\n")
                    }

                    append("=== USER REQUEST ===\n")
                    append(prompt)
                }

                val response = callOpenAIAPI(key, fullPrompt)

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
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

    /**
     * Calls OpenAI API with automatic endpoint selection.
     * First tries Chat Completions (v1/chat/completions). If the model is a base/completion model,
     * the API returns a 404 with a specific error message; then it retries with Completions (v1/completions).
     */
    private fun callOpenAIAPI(apiKey: String, prompt: String): String {
        log.debug("Starting API call to OpenAI with model: {}", selectedModel)

        // First attempt with chat completions
        try {
            return attemptChatCompletions(apiKey, prompt)
        } catch (e: Exception) {
            // Check if it's the specific "not a chat model" error
            if (e.message?.contains("not a chat model") == true || 
                (e is Exception && e.message?.contains("v1/completions") == true)) {
                log.debug("Model is not a chat model, retrying with completions endpoint")
                return attemptCompletions(apiKey, prompt)
            } else {
                // Some other error, rethrow
                throw e
            }
        }
    }

    private fun attemptChatCompletions(apiKey: String, prompt: String): String {
        log.debug("Attempting Chat Completions API with model: {}", selectedModel)

        val url = URL("https://api.openai.com/v1/chat/completions")
        val connection = url.openConnection() as HttpURLConnection

        try {
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $apiKey")
            connection.doOutput = true
            connection.connectTimeout = 30000
            connection.readTimeout = 30000

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

            log.debug("Chat request body: {}", requestBody.toString())

            connection.outputStream.use { os ->
                os.write(requestBody.toString().toByteArray())
            }

            val responseCode = connection.responseCode
            log.debug("Chat response code: {}", responseCode)

            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errorStream = connection.errorStream?.bufferedReader()?.readText() ?: "Unknown error"
                log.error("Chat error response: {}", errorStream)

                // Parse error response
                try {
                    val errorJson = JSONObject(errorStream)
                    val errorObj = errorJson.optJSONObject("error")
                    val errorMessage = errorObj?.optString("message") ?: errorStream
                    val errorType = errorObj?.optString("type") ?: ""
                    val errorCode = errorObj?.optString("code") ?: ""

                    log.error("Error type: {}, code: {}, message: {}", errorType, errorCode, errorMessage)

                    // If it's the "not a chat model" error, throw a specific exception to trigger retry
                    if (errorMessage.contains("not a chat model") || errorMessage.contains("v1/completions")) {
                        throw Exception("Not a chat model - retry with completions")
                    }

                    // Identify other specific error types
                    when {
                        responseCode == 429 || errorType.contains("rate_limit") || errorCode.contains("rate_limit") ->
                            throw com.tom.rv2ide.artificial.exceptions.RateLimitException("OpenAI rate limit exceeded: $errorMessage")
                        errorType.contains("insufficient_quota") || errorMessage.contains("quota") || errorMessage.contains("billing") ->
                            throw com.tom.rv2ide.artificial.exceptions.QuotaExceededException("OpenAI quota exceeded: $errorMessage")
                        errorType.contains("invalid_api_key") || errorCode.contains("invalid_api_key") ->
                            throw com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException("Invalid OpenAI API key: $errorMessage")
                        responseCode == 401 ->
                            throw com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException("OpenAI authentication failed: $errorMessage")
                        else ->
                            throw Exception("OpenAI API error ($responseCode) - Type: $errorType, Code: $errorCode, Message: $errorMessage")
                    }
                } catch (e: com.tom.rv2ide.artificial.exceptions.RateLimitException) {
                    throw e
                } catch (e: com.tom.rv2ide.artificial.exceptions.QuotaExceededException) {
                    throw e
                } catch (e: com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException) {
                    throw e
                } catch (e: Exception) {
                    // If we already wrapped it, rethrow; otherwise wrap
                    if (e.message == "Not a chat model - retry with completions") {
                        throw e
                    }
                    throw Exception("OpenAI API error ($responseCode): $errorStream")
                }
            }

            val responseBody = connection.inputStream.bufferedReader().readText()
            log.debug("Chat success response received, length: {}", responseBody.length)

            val jsonResponse = JSONObject(responseBody)

            val choices = jsonResponse.optJSONArray("choices")
            if (choices != null && choices.length() > 0) {
                val firstChoice = choices.getJSONObject(0)
                val message = firstChoice.optJSONObject("message")
                if (message != null) {
                    val content = message.optString("content")
                    if (content.isNotEmpty()) {
                        return content
                    }
                }
            }

            log.error("Failed to extract content from chat response. Full response: {}", responseBody.take(500))
            throw Exception("No response content from OpenAI Chat API. Response structure unexpected.")
        } catch (e: java.net.SocketTimeoutException) {
            log.error("Timeout exception", e)
            throw Exception("OpenAI request timeout: ${e.message}")
        } catch (e: java.net.UnknownHostException) {
            log.error("Network exception", e)
            throw Exception("Network error - cannot reach OpenAI: ${e.message}")
        } catch (e: Exception) {
            log.error("Chat attempt exception", e)
            throw e
        } finally {
            connection.disconnect()
        }
    }

    private fun attemptCompletions(apiKey: String, prompt: String): String {
        log.debug("Attempting Completions API with model: {}", selectedModel)

        // Combine system instructions and user prompt for base models
        val combinedPrompt = writingRules.useThis() + "\n\n" + prompt

        val url = URL("https://api.openai.com/v1/completions")
        val connection = url.openConnection() as HttpURLConnection

        try {
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $apiKey")
            connection.doOutput = true
            connection.connectTimeout = 30000
            connection.readTimeout = 30000

            val requestBody = JSONObject()
            requestBody.put("model", selectedModel)
            requestBody.put("prompt", combinedPrompt)
            requestBody.put("max_tokens", 4096)
            // Optionally set other parameters like temperature, stop sequences, etc.

            log.debug("Completions request body: {}", requestBody.toString())

            connection.outputStream.use { os ->
                os.write(requestBody.toString().toByteArray())
            }

            val responseCode = connection.responseCode
            log.debug("Completions response code: {}", responseCode)

            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errorStream = connection.errorStream?.bufferedReader()?.readText() ?: "Unknown error"
                log.error("Completions error response: {}", errorStream)

                // Parse error response
                try {
                    val errorJson = JSONObject(errorStream)
                    val errorObj = errorJson.optJSONObject("error")
                    val errorMessage = errorObj?.optString("message") ?: errorStream
                    val errorType = errorObj?.optString("type") ?: ""
                    val errorCode = errorObj?.optString("code") ?: ""

                    log.error("Error type: {}, code: {}, message: {}", errorType, errorCode, errorMessage)

                    // Identify specific error types
                    when {
                        responseCode == 429 || errorType.contains("rate_limit") || errorCode.contains("rate_limit") ->
                            throw com.tom.rv2ide.artificial.exceptions.RateLimitException("OpenAI rate limit exceeded: $errorMessage")
                        errorType.contains("insufficient_quota") || errorMessage.contains("quota") || errorMessage.contains("billing") ->
                            throw com.tom.rv2ide.artificial.exceptions.QuotaExceededException("OpenAI quota exceeded: $errorMessage")
                        errorType.contains("invalid_api_key") || errorCode.contains("invalid_api_key") ->
                            throw com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException("Invalid OpenAI API key: $errorMessage")
                        responseCode == 401 ->
                            throw com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException("OpenAI authentication failed: $errorMessage")
                        else ->
                            throw Exception("OpenAI API error ($responseCode) - Type: $errorType, Code: $errorCode, Message: $errorMessage")
                    }
                } catch (e: com.tom.rv2ide.artificial.exceptions.RateLimitException) {
                    throw e
                } catch (e: com.tom.rv2ide.artificial.exceptions.QuotaExceededException) {
                    throw e
                } catch (e: com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException) {
                    throw e
                } catch (e: Exception) {
                    throw Exception("OpenAI API error ($responseCode): $errorStream")
                }
            }

            val responseBody = connection.inputStream.bufferedReader().readText()
            log.debug("Completions success response received, length: {}", responseBody.length)

            val jsonResponse = JSONObject(responseBody)

            val choices = jsonResponse.optJSONArray("choices")
            if (choices != null && choices.length() > 0) {
                val firstChoice = choices.getJSONObject(0)
                val text = firstChoice.optString("text")
                if (text.isNotEmpty()) {
                    return text
                }
            }

            log.error("Failed to extract content from completions response. Full response: {}", responseBody.take(500))
            throw Exception("No response content from OpenAI Completions API. Response structure unexpected.")
        } catch (e: java.net.SocketTimeoutException) {
            log.error("Timeout exception", e)
            throw Exception("OpenAI request timeout: ${e.message}")
        } catch (e: java.net.UnknownHostException) {
            log.error("Network exception", e)
            throw Exception("Network error - cannot reach OpenAI: ${e.message}")
        } catch (e: Exception) {
            log.error("Completions attempt exception", e)
            throw e
        } finally {
            connection.disconnect()
        }
    }

    private fun readRelevantFiles(): Map<String, String> {
        val filesContent = mutableMapOf<String, String>()
        val tree = projectTreeResult?.tree ?: return filesContent

        val filePaths = tree.lines().filter { it.isNotBlank() }

        filePaths.forEach { filePath ->
            val trimmedPath = filePath.trim()
            val file = File(trimmedPath)

            if (file.isFile &&
                (trimmedPath.endsWith(".kt") ||
                        trimmedPath.endsWith(".java") ||
                        trimmedPath.endsWith(".xml") ||
                        trimmedPath.endsWith(".gradle") ||
                        trimmedPath.endsWith(".gradle.kts")) &&
                !trimmedPath.contains("/build/") &&
                !trimmedPath.contains("/.gradle/")) {
                try {
                    val content = file.readText()
                    filesContent[trimmedPath] = content
                } catch (e: Exception) {
                    // Skip files that can't be read
                }
            }
        }

        return filesContent
    }

    fun readFile(filePath: String): String? {
        return try {
            if (File(filePath).exists()) {
                File(filePath).readText()
            } else {
                projectTreeResult?.readFileContent(File(filePath).name)
            }
        } catch (e: Exception) {
            null
        }
    }

    override fun writeFile(filePath: String, content: String): FileWriteResult {
        val writer = fileWriter ?: return FileWriteResult.Error("File writer not initialized")
        return writer.writeFile(filePath, content, createBackup = true)
    }

    fun parseAndApplyModifications(response: String, capturedStates: Map<String, String>): List<FileModification> {
        val modifications = mutableListOf<FileModification>()
        val parser = com.tom.rv2ide.artificial.parser.SnippetParser()

        if (response.contains("FILE_TO_MODIFY:")) {
            val lines = response.lines()
            var currentFile: String? = null
            val contentBuilder = StringBuilder()
            var inContent = false

            for (line in lines) {
                if (line.startsWith("FILE_TO_MODIFY:")) {
                    if (currentFile != null && contentBuilder.isNotEmpty()) {
                        val rawContent = contentBuilder.toString().trim()
                        val cleanedContent = parser.cleanFileContent(rawContent)
                        val previousContent = capturedStates[currentFile]
                        val writeResult = writeFile(currentFile, cleanedContent)

                        val success = writeResult is FileWriteResult.Success
                        recordModification(currentFile, previousContent, cleanedContent, success)

                        modifications.add(FileModification(currentFile, cleanedContent, writeResult))
                    }

                    currentFile = line.substringAfter("FILE_TO_MODIFY:").trim()
                    contentBuilder.clear()
                    inContent = true
                } else if (inContent) {
                    contentBuilder.append(line).append("\n")
                }
            }

            if (currentFile != null && contentBuilder.isNotEmpty()) {
                val rawContent = contentBuilder.toString().trim()
                val cleanedContent = parser.cleanFileContent(rawContent)
                val previousContent = capturedStates[currentFile]
                val writeResult = writeFile(currentFile, cleanedContent)

                val success = writeResult is FileWriteResult.Success
                recordModification(currentFile, previousContent, cleanedContent, success)

                modifications.add(FileModification(currentFile, cleanedContent, writeResult))
            }
        }

        return modifications
    }

    override fun isInitialized(): Boolean = apiKey != null
}

data class FileModification(
    val filePath: String,
    val content: String,
    val writeResult: FileWriteResult
)

data class ConversationMessage(
    val role: String,
    val content: String
)