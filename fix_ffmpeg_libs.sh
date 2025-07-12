#!/usr/bin/env bash
set -e

echo "Fixing all missing FFmpeg dylib dependencies..."

while true; do
    # Try running ffmpeg
    OUT=$(ffmpeg -version 2>&1 || true)
    # Check for missing library error
    DYLIB=$(echo "$OUT" | grep -Eo '/usr/local/opt/[^ ]+\.dylib' | head -1)
    if [ -n "$DYLIB" ]; then
        echo "\nMissing library: $DYLIB"
        # Extract package name from path
        PKG=$(echo "$DYLIB" | sed -E 's#.*/opt/([^/]+)/lib/.*#\1#')
        DYLIB_FILE=$(basename "$DYLIB")
        echo "Homebrew package: $PKG"
        # Install the package
        brew install "$PKG" || true
        # Create the symlink if needed
        LIB_SRC="/usr/local/lib/$DYLIB_FILE"
        LIB_DST="/usr/local/opt/$PKG/lib/$DYLIB_FILE"
        sudo mkdir -p "/usr/local/opt/$PKG/lib"
        if [ -f "$LIB_SRC" ]; then
            echo "Symlinking $LIB_SRC -> $LIB_DST"
            sudo ln -sf "$LIB_SRC" "$LIB_DST"
        else
            # Try to find the dylib in the Cellar
            CELLAR_LIB=$(find /usr/local/Cellar/$PKG -name "$DYLIB_FILE" 2>/dev/null | head -1)
            if [ -n "$CELLAR_LIB" ]; then
                echo "Symlinking $CELLAR_LIB -> $LIB_DST"
                sudo ln -sf "$CELLAR_LIB" "$LIB_DST"
            else
                echo "Could not find $DYLIB_FILE for $PKG. Please check manually."
                exit 1
            fi
        fi
    else
        echo "\nAll FFmpeg dependencies resolved!"
        ffmpeg -version | head -10
        exit 0
    fi
done 