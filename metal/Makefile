# Metal kernel build rules
# xcrun metal -O3 → .air → .metallib

METALC   = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
FLAGS    = -O3 -ffast-math -std=metal3.1

SRCDIR   = .
BUILDDIR = build
TARGET   = $(BUILDDIR)/yunshu_kernels.metallib

SOURCES  = $(wildcard $(SRCDIR)/*.metal)
OBJECTS  = $(patsubst $(SRCDIR)/%.metal,$(BUILDDIR)/%.air,$(SOURCES))

.PHONY: all clean

all: $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.air: $(SRCDIR)/%.metal | $(BUILDDIR)
	$(METALC) $(FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(METALLIB) $^ -o $@

clean:
	rm -rf $(BUILDDIR)
