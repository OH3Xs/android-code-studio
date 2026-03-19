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
        // Hardcoded list of valid OpenAI model names (as of early 2025)
        private val VALID_MODELS = setOf(
            "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
            "o1-preview", "o1-mini", "gpt-4o-mini", "gpt-4.5-preview"
        )

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

            // Ensure we're using a valid OpenAI model
            if (!isValidModel(selectedModel)) {
                log.warn("Model '{}' is not in the valid OpenAI model list. Falling back to 'gpt-4o'", selectedModel)
                selectedModel = "gpt-4o"
                agents?.setAgent(selectedModel)
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

    private fun isValidModel(model: String): Boolean {
        return model in VALID_MODELS
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

                // Ensure model is valid before proceeding
                validateModel()

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
     * Ensures that the currently selected model is valid for OpenAI.
     * If not, falls back to a default model and updates the stored selection.
     */
    private fun validateModel() {
        if (!isValidModel(selectedModel)) {
            log.warn("Model '{}' is not valid for OpenAI. Falling back to 'gpt-4o'", selectedModel)
            selectedModel = "gpt-4o"
            agents?.setAgent(selectedModel)
            agents?.setProvider("openai")
        }
    }

    /**
     * Calls OpenAI's Responses API (v1/responses) which supports the latest models.
     */
    private fun callOpenAIAPI(apiKey: String, prompt: String): String {
        log.debug("Starting API call to OpenAI (Responses API) with model: {}", selectedModel)

        val url = URL("https://api.openai.com/v1/responses")
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
            requestBody.put("input", messages)
            // Temperature removed – some models (e.g., gpt-5-nano) do not support it
            requestBody.put("max_output_tokens", 4096)

            log.debug("Request body: {}", requestBody.toString())

            connection.outputStream.use { os ->
                os.write(requestBody.toString().toByteArray())
            }

            val responseCode = connection.responseCode
            log.debug("Response code: {}", responseCode)

            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errorStream = connection.errorStream?.bufferedReader()?.readText() ?: "Unknown error"
                log.error("Error response: {}", errorStream)

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
            log.debug("Success response received, length: {}", responseBody.length)

            val jsonResponse = JSONObject(responseBody)

            // Try to extract text from the response
            val output = jsonResponse.optJSONArray("output")
            if (output != null && output.length() > 0) {
                val firstOutput = output.getJSONObject(0)
                val contentArray = firstOutput.optJSONArray("content")
                if (contentArray != null && contentArray.length() > 0) {
                    val textBuilder = StringBuilder()
                    for (i in 0 until contentArray.length()) {
                        val contentItem = contentArray.getJSONObject(i)
                        val text = contentItem.optString("text")
                        if (text.isNotEmpty()) {
                            textBuilder.append(text)
                        }
                    }
                    if (textBuilder.isNotEmpty()) {
                        return textBuilder.toString()
                    }
                }
            }

            // If we couldn't extract text, log the response and throw
            log.error("Failed to extract content from response. Full response: {}", responseBody.take(500))
            throw Exception("No response content from OpenAI API. Response structure unexpected.")
        } catch (e: com.tom.rv2ide.artificial.exceptions.RateLimitException) {
            log.error("Rate limit exception", e)
            throw e
        } catch (e: com.tom.rv2ide.artificial.exceptions.QuotaExceededException) {
            log.error("Quota exceeded exception", e)
            throw e
        } catch (e: com.tom.rv2ide.artificial.exceptions.InvalidApiKeyException) {
            log.error("Invalid API key exception", e)
            throw e
        } catch (e: java.net.SocketTimeoutException) {
            log.error("Timeout exception", e)
            throw Exception("OpenAI request timeout: ${e.message}")
        } catch (e: java.net.UnknownHostException) {
            log.error("Network exception", e)
            throw Exception("Network error - cannot reach OpenAI: ${e.message}")
        } catch (e: Exception) {
            log.error("General exception", e)
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