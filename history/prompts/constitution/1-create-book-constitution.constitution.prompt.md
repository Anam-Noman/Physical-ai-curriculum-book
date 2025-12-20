---
id: 1
title: Create Book Constitution
stage: constitution
date_iso: 2025-01-08
surface: agent
model: Qwen
feature: ai-spec-driven-book
branch: main
user: user
command: /sp.constitution
labels:
  - constitution
  - spec-kit-plus
  - docusaurus
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - ".specify/memory/constitution.md"
tests_yaml: []
---

# Create Book Constitution

## Prompt Text

```
/sp.constitution /sp.constitution

Project: AI/Spec-Driven Book Creation and Deployment

Description:
Create a technical book using Docusaurus and Spec-Kit Plus, written with the assistance of Claude Code, and deployed via GitHub Pages. The book documents AI-native and spec-driven software development concepts in a clear, structured, and reproducible manner.

Core principles:
- Accuracy: All technical claims must be correct and verified against reliable sources
- Clarity: Content must be understandable to a technical audience with basic CS knowledge
- Spec-driven rigor: Follow Spec-Kit Plus structure strictly for consistency and traceability
- Reproducibility: Instructions, configs, and examples must be repeatable by readers
- Transparency: Clearly distinguish facts, assumptions, and examples

Key standards:
- All factual or technical claims must be supported by sources or official documentation
- Citation format: APA style (inline where applicable, references section at end)
- Source types: Official docs, peer-reviewed articles, and reputable technical blogs
- Plagiarism tolerance: 0% (all content must be original or properly cited)
- Code quality: All code examples must be syntactically correct and runnable
- Documentation style: Docusaurus-compatible MD/MDX only

Constraints:
- Structure must follow Spec-Kit Plus sections and conventions
- Each chapter must have a clear objective and summary
- Content must build incrementally from fundamentals to advanced topics
- Deployment instructions must work on GitHub Pages without modification

Success criteria:
- Book builds successfully with Docusaurus (no build or MDX errors)
- GitHub Pages deployment works correctly
- All claims are traceable and verifiable
- Code examples run without errors
- Content is clear, consistent, and technically accurate
```

## Response Text

Updated the project constitution at `.specify/memory/constitution.md` with the new principles for the AI/Spec-Driven Book Creation and Deployment project. The constitution now includes:

- Six core principles covering accuracy, clarity, spec-driven rigor, reproducibility, transparency, and code quality
- Key standards for citations, documentation style, and plagiarism tolerance
- Constraints and success criteria specific to the book project
- Governance guidelines for maintaining consistency

Added a Sync Impact Report indicating that dependent templates have been checked for consistency.

## Outcome

Successfully created and updated the project constitution file with the specific requirements for the AI/Spec-Driven Book Creation and Deployment project, ensuring that all principles, standards, constraints, and success criteria are clearly defined and aligned with the project goals.