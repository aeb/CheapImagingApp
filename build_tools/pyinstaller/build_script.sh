#!/bin/bash


BUILD_TOOLS_DIR="./build_tools/pyinstaller"
PYTHON_EXEC=python3.8

# Make sure that we are in the correct directory
if [ ! -d $BUILD_TOOLS_DIR ]; then
    echo "You must run this from your top project directory!"
    exit 1
fi

# Make sure that we have a main.spec with minimal bits
if [ -f ngehtapp.spec ]; then
    echo "WARNING: using existing ngehtapp.spec file."
else
    cp $BUILD_TOOLS_DIR/ngehtapp.spec .
fi

PYINSTALLER_ARTIFACTS="dist build pyinstaller"
for DIR in $PYINSTALLER_ARTIFACTS; do
    echo "Checking for ${DIR} ..." 
    if [ -d ${DIR} ]; then
	echo "ERROR: Found ./${DIR}, suggesting that a previous pyinstaller build exists."
	echo "       Please clean up before running again.  You need to remove"
	echo "       ${PYINSTALLER_ARTIFACTS}"
	exit 1
    fi
done

if [ -d build ]; then
    echo "ERROR: Found ./build, suggesting that a previous build exists.  Please clean up before running again."
    exist 1
fi

if [ -d pyinstaller ]; then
    echo "ERROR: Found ./, suggesting that a previous build exists.  Please clean up before running again."
    exist 1
fi



mkdir pyinstaller
pushd pyinstaller

# Run PyInstaller
$PYTHON_EXEC -m PyInstaller -y --clean --windowed ../ngehtapp.spec

pushd dist
hdiutil create ./ngehtapp.dmg -srcfolder ngehtapp.app -ov
popd

# create the dist dir for our result to be uploaded as an artifact
mkdir -p ../dist
cp "dist/ngEHTapp.dmg" ../dist/


