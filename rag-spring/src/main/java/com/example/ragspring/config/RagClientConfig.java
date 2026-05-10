package com.example.ragspring.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
@EnableConfigurationProperties(RagProperties.class)
public class RagClientConfig {

    @Bean
    public RestClient ragRestClient(RagProperties properties) {
        System.out.println("Python RAG baseUrl = " + properties.baseUrl());
        return RestClient.builder()
                .requestFactory(new SimpleClientHttpRequestFactory())
                .baseUrl(properties.baseUrl())
                .defaultHeader("X-API-Key", properties.apiKey())
                .build();
    }
}