# Changelog

All notable changes to AIGRC Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-13

### Added

#### Pydantic Schemas (AP-6)
- **AssetCard**: Complete AI asset registration with metadata
- **RiskFactors**: Risk assessment fields (autonomy level, data sensitivity, etc.)
- **Jurisdiction**: Regulatory jurisdiction tracking
- **GoldenThread**: Traceability link to authorizing ticket
- **CapabilitiesManifest**: Runtime capability constraints
- Full TypeScript schema compatibility

#### Asset Card Management (AP-7)
- **CRUD Operations**: Create, read, update, delete asset cards
- **YAML Serialization**: Human-readable asset card storage
- **Validation**: Pydantic validation with detailed error messages
- **File Operations**: Load/save asset cards from filesystem

#### Golden Thread (AP-11)
- **Hash Computation**: SHA-256 hash generation for asset cards
- **TypeScript Compatibility**: Identical hash output to TypeScript implementation
- **Verification**: Hash verification for integrity checking

#### Detection Engine (AP-13)
- **Scanner**: Codebase scanning for AI/ML patterns
- **PatternRegistry**: Extensible pattern matching system
- **Framework Detection**: LangChain, AutoGen, CrewAI, OpenAI, Anthropic patterns
- **Risk Assessment**: Automatic risk factor inference from detected patterns

#### CLI (AP-20)
- **aigrc scan**: Scan codebase for AI agents
- **aigrc init**: Initialize new asset card
- **aigrc register**: Register asset with governance platform
- **aigrc status**: Check registration status
- **aigrc validate**: Validate asset card schema
- **aigrc hash**: Compute Golden Thread hash
- Built with Typer for excellent CLI UX

### Dependencies
- pydantic >= 2.0
- pyyaml >= 6.0
- typer >= 0.9

### Optional Dependencies
- `[dev]`: pytest, pytest-cov, mypy, ruff, black

## Links
- [GitHub Repository](https://github.com/aigrc/aigrc-python)
- [Documentation](https://aigrc.readthedocs.io)
- [Issue Tracker](https://github.com/aigrc/aigrc-python/issues)
