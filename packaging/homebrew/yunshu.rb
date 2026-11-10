# Homebrew formula for the shared YuhuanStudio tap
# (https://github.com/YuhuanStudio/homebrew-tap, file Formula/yunshu.rb).
# Install: brew install yuhuanstudio/tap/yunshu
#
# The source is the PyPI sdist of the same version; the sha256 is of the file PyPI serves.
#
# Background running is `yunshu service install` (one launchd agent for every
# install method), so this formula has no `service do` block.
class Yunshu < Formula
  desc "Fast local LLM/VLM inference engine for Apple Silicon (MLX)"
  homepage "https://github.com/YuhuanStudio/Yunshu"
  url "https://files.pythonhosted.org/packages/54/ed/e863fe4b4673c67ab7d319d9c64c56538fb221abe2aed3c7bedcd27b0238/yunshu-0.1.7.tar.gz"
  sha256 "ad7c5c07bd6ce8779e57269c37bc079b45d51ea4f6101698c6f355406950b869"
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
