package com.example.ragspring.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "rag")
public record RagProperties(
        String baseUrl,
        String apiKey
) {
}