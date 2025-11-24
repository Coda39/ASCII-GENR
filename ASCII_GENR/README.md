# ASCII Art Generator

Command-line tool for generating ASCII art from images with edge detection and optional color support.

## Usage

### Basic Usage

```bash
python3 ascii_generation.py <image_path>
```

### Options

**Color Output**

```bash
python3 ascii_generation.py image.png --color
```

**Rasterized Image Output**

```bash
python3 ascii_generation.py image.png --render
```

**Custom Font Size**

```bash
python3 ascii_generation.py image.png --render --font-size 30
```

**Custom Tile Size**

```bash
python3 ascii_generation.py image.png --tile-size 16
```

**Adjust Edge Detection**

```bash
python3 ascii_generation.py image.png --edge-threshold 12
```

**Skip Debug Visualizations**

```bash
python3 ascii_generation.py image.png --no-debug
```

### Combined Options

```bash
python3 ascii_generation.py image.png --color --render --font-size 25 --tile-size 16 --no-debug
```

## CLI Arguments

- `image_path` - Path to input image (required)
- `--color` - Enable colored ASCII output (default: monochrome)
- `--render` - Generate rasterized PNG image of ASCII art (default: text only)
- `--font-size` - Font size for rendered image in pixels (default: 20)
- `--tile-size` - ASCII character tile size (default: 8)
- `--edge-threshold` - Minimum edge pixels for edge detection (default: 12)
- `--no-debug` - Skip generating debug visualizations

## Output Files

- `ascii_art.txt` - Plain text ASCII art
- `ascii_art_color.txt` - ANSI colored ASCII art (if `--color` used)
- `ascii_art_rendered.png` - Rasterized image (if `--render` used)
- `debug_*.png` - Debug visualizations (unless `--no-debug` used)
