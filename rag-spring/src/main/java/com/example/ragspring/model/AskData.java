package com.example.ragspring.model;

import java.util.List;

public record AskData(
        String answer,
        List<SourceItem> sources
) {
}