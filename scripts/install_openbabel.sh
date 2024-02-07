# install cmake if not installed
if ! [ -x "$(command -v cmake)" ]; then
  sudo apt-get update
  sudo apt-get install -y cmake
fi

wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1-source.tar.bz2
tar -jxvf openbabel-3.1.1-source.tar.bz2
cd openbabel-3.1.1
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/Tools/openbabel-install
make -j4
make install

# add to path
echo 'export PATH=$PATH:~/Tools/openbabel-install/bin' >> ~/.bashrc
source ~/.bashrc

# clean up
cd ../..
rm -rf openbabel-3.1.1 openbabel-3.1.1-source.tar.bz2

# Remove the /Tools/openbabel-install directory if you want to uninstall openbabel
# and remove the PATH addition from ~/.bashrc