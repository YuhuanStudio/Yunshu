// YunshuSDK — Swift client for the Yunshu inference platform.
// OpenAI-compatible API on Apple Silicon clusters.

import Foundation

// MARK: - Errors

/// Errors thrown by ``YunshuClient``.
public enum YunshuError: Error, LocalizedError, Equatable {
    /// A network-level failure (no response, timeout, DNS, etc.).
    case networkError(String)
    /// The server returned a non-2xx status with an error message.
    case apiError(String)
    /// The response body could not be decoded.
    case decodingError
    /// Authentication failure (missing or invalid API key).
    case authError

    public var errorDescription: String? {
        switch self {
        case .networkError(let msg): return "Network error: \(msg)"
        case .apiError(let msg):     return "API error: \(msg)"
        case .decodingError:         return "Failed to decode server response"
        case .authError:             return "Authentication failed"
        }
    }
}

// MARK: - Response Models

/// A single chat completion message.
public struct ChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// A single choice within a ``ChatResponse``.
public struct ChatChoice: Codable, Equatable, Sendable {
    public let index: Int
    public let message: ChatMessage
    public let finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
    }
}

/// Response from `/v1/chat/completions`.
public struct ChatResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [ChatChoice]
    public let usage: Usage?
}

/// A single choice within a ``CompletionResponse``.
public struct CompletionChoice: Codable, Equatable, Sendable {
    public let index: Int
    public let text: String
    public let finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }
}

/// Response from `/v1/completions`.
public struct CompletionResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [CompletionChoice]
    public let usage: Usage?
}

/// A single embedding within an ``EmbeddingResponse``.
public struct EmbeddingData: Codable, Equatable, Sendable {
    public let object: String
    public let index: Int
    public let embedding: [Double]
}

/// Response from `/v1/embeddings`.
public struct EmbeddingResponse: Codable, Equatable, Sendable {
    public let object: String
    public let data: [EmbeddingData]
    public let model: String
    public let usage: Usage?
}

/// Token usage statistics returned alongside completion requests.
public struct Usage: Codable, Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

/// Metadata about a model registered on the server.
public struct ModelInfo: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let ownedBy: String?

    private enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

/// The top-level list response from `/v1/models`.
private struct ModelListResponse: Codable {
    let data: [ModelInfo]
}

// MARK: - Client

/// Swift client for a Yunshu inference server.
///
/// Usage:
/// ```swift
/// let client = YunshuClient(baseURL: "http://localhost:8000")
/// let reply = try await client.chat(
///     messages: [("user", "Hello!")],
///     model: "qwen2.5-0.5b-instruct"
/// )
/// print(reply.choices.first?.message.content ?? "")
/// ```
public final class YunshuClient: Sendable {

    /// Base URL of the Yunshu server (no trailing slash).
    public let baseURL: URL

    /// Optional API key sent as `Authorization: Bearer <key>`.
    public let apiKey: String?

    private let session: URLSession

    /// Create a new client.
    ///
    /// - Parameters:
    ///   - baseURL: Scheme + host + port, e.g. `http://localhost:8000`.
    ///   - apiKey:  Optional bearer token for authenticated endpoints.
    public init(baseURL: String, apiKey: String? = nil) {
        let trimmed = baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.baseURL = URL(string: trimmed)!
        self.apiKey = apiKey

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)
    }

    // MARK: Chat Completions

    /// Send a chat completion request to `/v1/chat/completions`.
    ///
    /// - Parameters:
    ///   - messages:    Array of `(role, content)` tuples.
    ///   - model:       Model ID (omit to use the server default).
    ///   - temperature: Sampling temperature.
    /// - Returns: A ``ChatResponse`` with generated messages.
    public func chat(
        messages: [(role: String, content: String)],
        model: String? = nil,
        temperature: Double? = nil
    ) async throws -> ChatResponse {
        var body: [String: Any] = [
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
        ]
        if let model { body["model"] = model }
        if let temperature { body["temperature"] = temperature }

        return try await request(.POST, "/v1/chat/completions", body: body)
    }

    // MARK: Text Completions

    /// Send a text completion request to `/v1/completions`.
    ///
    /// - Parameters:
    ///   - prompt:    The text prompt.
    ///   - model:     Model ID (omit to use the server default).
    ///   - maxTokens: Maximum tokens to generate.
    /// - Returns: A ``CompletionResponse`` with generated text.
    public func completions(
        prompt: String,
        model: String? = nil,
        maxTokens: Int? = nil
    ) async throws -> CompletionResponse {
        var body: [String: Any] = ["prompt": prompt]
        if let model { body["model"] = model }
        if let maxTokens { body["max_tokens"] = maxTokens }

        return try await request(.POST, "/v1/completions", body: body)
    }

    // MARK: Embeddings

    /// Generate embeddings for the given text via `/v1/embeddings`.
    ///
    /// - Parameters:
    ///   - text:  Input text to embed.
    ///   - model: Embedding model ID.
    /// - Returns: An ``EmbeddingResponse`` with embedding vectors.
    public func embeddings(
        text: String,
        model: String? = nil
    ) async throws -> EmbeddingResponse {
        var body: [String: Any] = ["input": text]
        if let model { body["model"] = model }

        return try await request(.POST, "/v1/embeddings", body: body)
    }

    // MARK: Models

    /// List all models registered on the server.
    ///
    /// Calls `GET /v1/models` and returns the model metadata array.
    public func listModels() async throws -> [ModelInfo] {
        let wrapper: ModelListResponse = try await request(.GET, "/v1/models")
        return wrapper.data
    }

    // MARK: - Internal

    private enum HTTPMethod: String {
        case GET, POST
    }

    private func request<T: Decodable>(
        _ method: HTTPMethod,
        _ path: String,
        body: [String: Any]? = nil
    ) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        if let body {
            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            } catch {
                throw YunshuError.networkError("Failed to encode request body")
            }
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw YunshuError.networkError(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw YunshuError.networkError("Unexpected response type")
        }

        if http.statusCode == 401 || http.statusCode == 403 {
            throw YunshuError.authError
        }

        guard (200...299).contains(http.statusCode) else {
            let message = parseErrorMessage(data) ?? "HTTP \(http.statusCode)"
            throw YunshuError.apiError(message)
        }

        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw YunshuError.decodingError
        }
    }

    /// Attempt to extract a human-readable error message from a JSON error body.
    private func parseErrorMessage(_ data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        // OpenAI-style: {"error": {"message": "..."}}
        if let error = obj["error"] as? [String: Any],
           let message = error["message"] as? String {
            return message
        }
        // Flat: {"detail": "..."}
        if let detail = obj["detail"] as? String {
            return detail
        }
        return nil
    }
}
