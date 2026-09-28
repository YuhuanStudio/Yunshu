# Homebrew formula for the shared YuhuanStudio tap
# (https://github.com/YuhuanStudio/homebrew-tap, file Formula/yunshu.rb).
# Install: brew install yuhuanstudio/tap/yunshu
#
# The source is the PyPI sdist of the same version. The sha256 below is of a
# local `uv build` of 0.1.1; the file PyPI serves is built by the release
# workflow, so replace it after publishing (packaging/homebrew/README.md).
#
# Background running is `yunshu service install` (one launchd agent for every
# install method), so this formula has no `service do` block.
class Yunshu < Formula
  desc "Fast local LLM/VLM inference engine for Apple Silicon (MLX)"
  homepage "https://github.com/YuhuanStudio/Yunshu"
  url "https://files.pythonhosted.org/packages/source/y/yunshu/yunshu-0.1.1.tar.gz"
  sha256 "01cfa3e9c80be6862e6c069bdbf0d5cfacde9add3cf7e5358ddd3c6e49ad7dd8"
  license "Apache-2.0"
  head "https://github.com/YuhuanStudio/Yunshu.git", branch: "main"

  livecheck do
    url :stable
    strategy :pypi
  end

  depends_on arch: :arm64
  depends_on macos: :sonoma
  depends_on "python@3.13"

  def install
    system "python3.13", "-m", "venv", libexec
    # The vision extra (mlx-vlm) is what the Qwen3.5 / 3.6 / 3.8 family and
    # every VLM need; the other extras stay opt-in via pip in libexec.
    system libexec/"bin/pip", "install", "#{buildpath}[vision]"
    bin.install_symlink libexec/"bin/yunshu"
  end

  def caveats
    <<~EOS
      Check the machine and download a model:
        yunshu doctor
        yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit

      Serve it (http://127.0.0.1:8000/v1):
        yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit

      Run it in the background at login:
        yunshu service install -m mlx-community/Qwen3.5-9B-MLX-4bit

      Models live in ~/.yunshu/models (change it with
      `yunshu config set models_dir <path>`); models already in the
      Hugging Face cache are used in place.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/yunshu --version")
  end
end
