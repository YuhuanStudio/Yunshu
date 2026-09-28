# Draft Homebrew formula. Not published: it goes into a tap once the tap's
# location is decided. Fill in url/sha256 from the release tarball when a
# version is tagged (RELEASING.md, "After the release").
#
# Background running is `yunshu service install` (one launchd agent for every
# install method), so this formula has no `service do` block.
class Yunshu < Formula
  desc "Fast local LLM/VLM inference engine for Apple Silicon (MLX)"
  homepage "https://github.com/YuhuanStudio/Yunshu"
  url "https://github.com/YuhuanStudio/Yunshu/archive/refs/tags/v0.1.1.tar.gz"
  sha256 "FILL_IN_AT_RELEASE"
  license "Apache-2.0"
  head "https://github.com/YuhuanStudio/Yunshu.git", branch: "main"

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
        yunshu serve -m ~/.yunshu/models/mlx-community/Qwen3.5-9B-MLX-4bit

      Run it in the background at login:
        yunshu service install -m ~/.yunshu/models/mlx-community/Qwen3.5-9B-MLX-4bit
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/yunshu --version")
  end
end
