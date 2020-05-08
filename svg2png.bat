cd libraries/inkscape/bin
set arg1=%1%
set arg2=%2%

echo %arg1%
echo %arg2%

set path1=%arg1%
set path2=%arg2%

inkscape  %path1% -o %path2%

