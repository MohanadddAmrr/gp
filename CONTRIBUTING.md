# Contributing to TactiVision Pro

Thank you for your interest in contributing to TactiVision Pro! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Fork the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/tactivision-pro.git
   cd tactivision-pro
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original/tactivision-pro.git
   ```

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

### Run Setup

```bash
python main.py --mode setup
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please:

1. Check if the issue already exists
2. Update to the latest version to see if it's been fixed

When reporting bugs, include:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Relevant code snippets or error messages

### Suggesting Features

Feature requests are welcome! Please provide:

- Clear description of the feature
- Use cases and benefits
- Possible implementation approach (optional)

### Contributing Code

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Changes**
   - Write clear, documented code
   - Follow coding standards
   - Add tests for new functionality

3. **Test Your Changes**
   ```bash
   python -m pytest tests/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Follow conventional commits format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding tests
   - `refactor:` Code refactoring
   - `style:` Formatting changes
   - `chore:` Maintenance tasks

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters max
- Use type hints where possible
- Docstrings for all public functions/classes

### Code Formatting

We use the following tools:

```bash
# Format code
black services/ tests/ demo/

# Sort imports
isort services/ tests/ demo/

# Check style
flake8 services/ tests/ demo/

# Type checking
mypy services/
```

### Documentation Strings

Use Google-style docstrings:

```python
def process_video(video_path: str, output_dir: str) -> bool:
    """Process a video file for analysis.
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory for output files.
        
    Returns:
        True if processing succeeded, False otherwise.
        
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If output_dir is not a valid directory.
        
    Example:
        >>> success = process_video("match.mp4", "./outputs")
        >>> print(f"Processing {'succeeded' if success else 'failed'}")
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_ball_tracker.py

# Run with coverage
python -m pytest tests/ --cov=services --cov-report=html

# Run performance benchmarks
python tests/test_suite.py --benchmark
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

Example:

```python
def test_ball_tracker_update():
    """Test that ball tracker correctly updates position."""
    # Arrange
    tracker = BallTracker()
    position = (100, 200)
    
    # Act
    tracker.update(position, 0.9)
    
    # Assert
    assert tracker.get_smoothed_position() == position
```

## Documentation

### Code Documentation

- Document all public APIs
- Include type hints
- Provide usage examples

### User Documentation

- Update README.md for user-facing changes
- Add entries to CHANGELOG.md
- Update relevant docs in `docs/` folder

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add changelog entry
   - Rebase on latest main branch

2. **PR Description**
   - Clear title following conventional commits
   - Description of changes
   - Related issue numbers
   - Screenshots (if UI changes)

3. **Review Process**
   - Maintainers will review within 48 hours
   - Address review comments
   - Keep discussion focused and professional

4. **After Merge**
   - Delete your branch
   - Update your local main branch

## Development Guidelines

### Adding New Services

1. Create module in `services/`
2. Add comprehensive docstrings
3. Write unit tests
4. Update main.py if CLI integration needed
5. Add to dashboard if UI needed

### Database Changes

1. Update `services/database_schema.py`
2. Create migration script
3. Test migration on sample data
4. Document changes

### UI Changes

1. Follow existing design patterns
2. Test on different screen sizes
3. Ensure accessibility
4. Add tooltips for complex features

## Questions?

- Join our [Discord community](https://discord.gg/tactivision)
- Open a [GitHub Discussion](https://github.com/yourusername/tactivision-pro/discussions)
- Email: dev@tactivision.pro

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to TactiVision Pro! ⚽
