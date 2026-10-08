# Homebrew

Yunshu ships in the shared YuhuanStudio tap,
[YuhuanStudio/homebrew-tap](https://github.com/YuhuanStudio/homebrew-tap). The tap already holds the
YunAudio cask. `yunshu.rb` here is the formula; in the tap it goes to `Formula/yunshu.rb`.

```sh
brew install yuhuanstudio/tap/yunshu
```

## Adding or updating the formula (maintainer)

Do this after the version is on PyPI, since the formula installs the PyPI sdist.

```sh
# 1. The sha256 of the sdist PyPI serves. This is the file the release
#    workflow built, not a local build, so check it every release.
V=0.1.1
curl -sL "https://files.pythonhosted.org/packages/source/y/yunshu/yunshu-$V.tar.gz" | shasum -a 256

# 2. Put that value into packaging/homebrew/yunshu.rb (url and sha256 for $V)
#    and commit it in this repo.

# 3. Copy the formula into the tap and push.
git clone https://github.com/YuhuanStudio/homebrew-tap.git
cd homebrew-tap
mkdir -p Formula
cp ../Yunshu/packaging/homebrew/yunshu.rb Formula/yunshu.rb
brew style Formula/yunshu.rb
git add Formula/yunshu.rb
git commit -m "yunshu $V"
git push

# 4. Check it from the tap.
brew update
brew install yuhuanstudio/tap/yunshu
brew test yunshu
yunshu doctor
```

The tap's README lists what it installs. Add a line for Yunshu next to YunAudio:

```sh
brew install yuhuanstudio/tap/yunshu
```

## CLI UX proposal

`cliux-proposed.diff` proposes completion installation and `yunshu setup` for the
next published release containing those commands. The current tap formula is
0.1.4; do not apply the proposal to that sdist. Update URL/SHA from PyPI when
the new version is published, then review/apply the diff in the tap. This worker
has not edited or pushed the tap.
