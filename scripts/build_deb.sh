#!/bin/bash

# Simple script to create a .deb package for sibi-platform
PACKAGE_NAME="sibi-platform"
VERSION="1.0.0"
MAINTAINER="Vicosilalahi <vico@example.com>"
DESCRIPTION="Unified SIBI Sign Language Recognition Platform"

STAGING_DIR="dist/deb_staging"

# Create directory structure
mkdir -p $STAGING_DIR/DEBIAN
mkdir -p $STAGING_DIR/opt/$PACKAGE_NAME
mkdir -p $STAGING_DIR/usr/bin

# Create control file
cat <<EOF > $STAGING_DIR/DEBIAN/control
Package: $PACKAGE_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: amd64
Maintainer: $MAINTAINER
Description: $DESCRIPTION
Depends: python3, python3-tk, python3-pil.imagetk, portaudio19-dev
EOF

# Copy source files (excluding large datasets for this skeleton)
cp -r app piper scripts actions.json sibi.py $STAGING_DIR/opt/$PACKAGE_NAME/

# Create symlink
ln -s /opt/$PACKAGE_NAME/sibi.py $STAGING_DIR/usr/bin/sibi

# Build package
dpkg-deb --build $STAGING_DIR dist/${PACKAGE_NAME}_${VERSION}_amd64.deb

echo "Debian package created in dist/"
