import nox


@nox.session(python=["3.12"])
def docs(session: nox.Session):
    """Build documentation using the same steps as CI."""
    # Upgrade pip and install uv (matching CI setup)
    session.run("python", "-m", "pip", "install", "--upgrade", "pip")
    session.run("python", "-m", "pip", "install", "--upgrade", "uv")

    # Install pytest (as done in CI)
    session.run("python", "-m", "uv", "pip", "install", "pytest")

    # Install the three packages (matching CI order) - adjust paths to go up one level
    session.run("python", "-m", "uv", "pip", "install", "../cuequivariance")
    session.run("python", "-m", "uv", "pip", "install", "../cuequivariance_jax")
    session.run("python", "-m", "uv", "pip", "install", "../cuequivariance_torch")

    # Explicitly install jax-triton (workaround for dependency resolution issue)
    session.run("python", "-m", "pip", "install", "jax-triton")

    # Install docs requirements - now in same directory
    session.run("python", "-m", "uv", "pip", "install", "-r", "requirements.txt")

    # Build sphinx documentation - adjust paths (current dir is docs, output to public)
    session.run("sphinx-build", "-b", "html", ".", "public")
