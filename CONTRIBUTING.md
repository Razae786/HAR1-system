# Contributing to HAR1-System

Thank you for considering contributing to the Human Activity Recognition system! This document provides guidelines and instructions for contributing.

---

## 🤝 Ways to Contribute

- **Report Bugs**: Submit issues with clear descriptions and reproduction steps
- **Suggest Features**: Propose new activity classes or detection methods
- **Improve Documentation**: Fix typos, add examples, or clarify explanations
- **Optimize Performance**: Profile code and submit performance improvements
- **Add Tests**: Write unit tests to improve code reliability
- **Submit Code**: Implement features or bug fixes with pull requests

---

## 🐛 Reporting Bugs

Before creating a bug report, please check if the issue already exists. When reporting bugs, include:

- **Title**: Clear, descriptive title
- **Description**: What happened vs. what you expected
- **Reproduction Steps**: Step-by-step instructions to reproduce
- **Environment**: Python version, OS, key dependencies
- **Error Message**: Full traceback if applicable
- **Screenshots/Video**: Visual evidence if relevant

### Example Bug Report

```
Title: Model predictions incorrect on rotated pose

Description:
When testing with rotated video angles, the classifier returns "No Pose" 
even though landmarks are clearly visible.

Steps to Reproduce:
1. Upload video with 45° camera angle
2. Run pose detection
3. Observe "No Pose" predictions

Environment:
- Python 3.9.0
- MediaPipe 0.10.0
- scikit-learn 1.3.0
```

---

## 💡 Suggesting Features

Features should follow the format:

```
Title: [FEATURE] Brief description

Problem Statement:
Describe the current limitation or use case not supported.

Proposed Solution:
How would you solve this?

Alternative Approaches:
Are there other ways to implement this?

Additional Context:
Any relevant information, diagrams, or references.
```

---

## 🔄 Pull Request Process

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HAR1-system.git
   cd HAR1-system
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make Changes**
   - Follow PEP 8 style guidelines
   - Add docstrings to functions
   - Write clear commit messages
   - Include type hints where possible

4. **Test Your Changes**
   ```bash
   python -m pytest tests/
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add brief description of changes"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to GitHub and create a PR
   - Use the PR template below
   - Link related issues
   - Describe changes clearly

---

## 📝 Pull Request Template

```markdown
## Description
Brief description of changes and why they're needed.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Related Issues
Closes #(issue number)

## How Has This Been Tested?
Describe the testing approach:
- [ ] Unit tests
- [ ] Manual testing
- [ ] Benchmark comparison

## Changes Checklist
- [ ] Code follows PEP 8 style
- [ ] Comments and docstrings added
- [ ] No new warnings generated
- [ ] Existing tests pass
- [ ] New tests added (if applicable)

## Performance Impact
- [ ] No performance impact
- [ ] Improvement: X% faster (with benchmarks)
- [ ] Breaking change: requires model retraining

## Screenshots/Videos (if applicable)
Add before/after comparisons or demo videos.
```

---

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/HAR1-system.git
cd HAR1-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Format code
black streamlit_app.py app.py

# Check style
flake8 streamlit_app.py

# Run tests
pytest tests/
```

---

## 🎨 Code Style Guidelines

### Python Style (PEP 8)
```python
# ✅ Good
def calculate_angle(point_a, point_b, point_c):
    """Calculate angle between three points.
    
    Args:
        point_a: First point coordinates [x, y]
        point_b: Vertex point coordinates [x, y]
        point_c: Third point coordinates [x, y]
    
    Returns:
        float: Angle in degrees
    """
    vector_ba = np.array(point_b) - np.array(point_a)
    vector_bc = np.array(point_c) - np.array(point_b)
    return np.degrees(np.arccos(...))

# ❌ Bad
def calc_angle(a,b,c):
    # calculates angle
    return degrees(arccos(...))
```

### Naming Conventions
- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_`

### Comments
```python
# ✅ Good: explains WHY
# Skip every 2nd frame to reduce latency (target: 30 FPS)
if frame_count % 2 == 0:
    process_frame()

# ❌ Bad: states the obvious
# Check if frame_count is divisible by 2
if frame_count % 2 == 0:
```

---

## 📚 Documentation

When adding features, update documentation:

1. **Docstrings**: All public functions and classes
2. **README.md**: New features in features section
3. **OPTIMIZATION.md**: Performance-related changes
4. **Code Comments**: Complex logic that isn't self-explanatory

---

## 🧪 Testing Guidelines

Write tests for:
- Core functions (angle calculation, feature extraction)
- Edge cases (empty landmarks, invalid coordinates)
- Integration (full pipeline)

```python
# tests/test_pose_utils.py
import pytest
import numpy as np
from streamlit_app import calculate_angle

def test_calculate_angle_basic():
    """Test angle calculation with known values."""
    # 90 degree angle
    a = [0, 0]
    b = [1, 0]
    c = [1, 1]
    angle = calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1.0

def test_calculate_angle_invalid_input():
    """Test with invalid input."""
    with pytest.raises((ValueError, TypeError)):
        calculate_angle([0, 0], [1, 1], None)
```

---

## 🚀 Performance Contribution

If submitting a performance improvement:

1. **Benchmark Before/After**
   ```python
   import timeit
   
   # Before
   old_time = timeit.timeit(old_function, number=1000)
   
   # After
   new_time = timeit.timeit(new_function, number=1000)
   
   improvement = (1 - new_time/old_time) * 100
   print(f"Improvement: {improvement:.1f}%")
   ```

2. **Include Metrics**: FPS, latency, memory usage
3. **Test Compatibility**: Ensure it works on CPU (no GPU dependency)
4. **Document Tradeoffs**: Any accuracy loss or limitations?

---

## 📞 Questions or Need Help?

- **GitHub Discussions**: Ask questions in issues
- **Documentation**: Check README.md and docs/
- **Email**: Open an issue marked as "question"

---

## ✨ Code Review Process

Maintainers will review PRs for:

1. **Functionality**: Does it work correctly?
2. **Code Quality**: Is it maintainable and well-structured?
3. **Performance**: Does it improve or maintain performance?
4. **Testing**: Is it adequately tested?
5. **Documentation**: Is it documented?

Expect 1-3 rounds of feedback. We appreciate your patience!

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🎉**
