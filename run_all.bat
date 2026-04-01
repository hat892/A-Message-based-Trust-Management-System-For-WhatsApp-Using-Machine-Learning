@echo off
echo ========================================
echo  Running All Experiments
echo ========================================
echo.

echo [1/2] Running EigenTrust - Naive Strategy...
for %%f in (*.trace) do (
    echo   Running EigenTrust naive on %%f
    java -cp "." TraceSimulator -input %%f -tm eigen -strategy naive
)

echo.
echo [2/2] Running WhatsTrust+URL - Naive Strategy...
for %%f in (*.trace) do (
    echo   Running WhatsTrust naive on %%f
    java -cp "." TraceSimulator -input %%f -tm whatstrust -strategy naive
)

echo.
echo ========================================
echo  All experiments complete!
echo  Now run: python parse_results.py
echo ========================================
pause
