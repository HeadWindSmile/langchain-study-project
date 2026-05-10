package com.example.ragspring.model;

public record ApiResponse<T>(
        Integer code,
        String message,
        T data,
        String requestId
) {
}