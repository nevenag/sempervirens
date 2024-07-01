#! /usr/bin/sh

mkdir -p /tmp/fp_0_001
mkdir -p /tmp/me_0_1/fp_0_001
mkdir -p /tmp/me_0_15/fp_0_001

mkdir -p /tmp/fp_0_01
mkdir -p /tmp/me_0_1/fp_0_01
mkdir -p /tmp/me_0_15/fp_0_01

echo ""
echo "Higher MEP"

# echo ""
# echo "MEP 0.05"
# echo ""
# echo "FPP 0.001"
# poetry run fpp0_001_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_001_1000x1000s --algorithm huntress
# poetry run fpp0_001_1000x1000s --algorithm scistreep
# poetry run fpp0_001_1000x1000s --algorithm scite
# echo ""
# echo "FPP 0.01"
# poetry run fpp0_01_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_01_1000x1000s --algorithm huntress
# poetry run fpp0_01_1000x1000s --algorithm scistreep
# poetry run fpp0_01_1000x1000s --algorithm scite

echo ""
echo "MEP 0.1"
echo ""
echo "FPP 0.001"
poetry run fpp0_001_0_1mer_1000x1000s --algorithm sempervirens_rs
poetry run fpp0_001_0_1mer_1000x1000s --algorithm huntress
poetry run fpp0_001_0_1mer_1000x1000s --algorithm scistreep
poetry run fpp0_001_0_1mer_1000x1000s --algorithm scite
echo ""
echo "FPP 0.01"
poetry run fpp0_01_0_1mer_1000x1000s --algorithm sempervirens_rs
poetry run fpp0_01_0_1mer_1000x1000s --algorithm huntress
poetry run fpp0_01_0_1mer_1000x1000s --algorithm scistreep
poetry run fpp0_01_0_1mer_1000x1000s --algorithm scite

echo ""
echo "MEP 0.15"
echo ""
echo "FPP 0.001"
poetry run fpp0_001_0_15mer_1000x1000s --algorithm sempervirens_rs
poetry run fpp0_001_0_15mer_1000x1000s --algorithm huntress
poetry run fpp0_001_0_15mer_1000x1000s --algorithm scistreep
poetry run fpp0_001_0_15mer_1000x1000s --algorithm scite
echo ""
echo "FPP 0.01"
poetry run fpp0_01_0_15mer_1000x1000s --algorithm sempervirens_rs
poetry run fpp0_01_0_15mer_1000x1000s --algorithm huntress
poetry run fpp0_01_0_15mer_1000x1000s --algorithm scistreep
poetry run fpp0_01_0_15mer_1000x1000s --algorithm scite


echo ""
echo ""
echo ""
echo ""
echo "Robustness to errors in FPP and FNP"

echo ""
echo "FPP"

echo ""
echo "1000x1000s robustness to FPP, true value is 0.001"
# echo ""
# echo "plug in true"
# poetry run fpp0_001_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_001_1000x1000s --algorithm huntress
# poetry run fpp0_001_1000x1000s --algorithm scistreep
# poetry run fpp0_001_1000x1000s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.0001"
poetry run fpp0_001_1000x1000s --algorithm sempervirens_rs --fpp_override 0.0001
poetry run fpp0_001_1000x1000s --algorithm huntress --fpp_override 0.0001
poetry run fpp0_001_1000x1000s --algorithm scistreep --fpp_override 0.0001
poetry run fpp0_001_1000x1000s --algorithm scite --fpp_override 0.0001
echo ""
echo "plug in 10x -> 0.01"
poetry run fpp0_001_1000x1000s --algorithm sempervirens_rs --fpp_override 0.01
poetry run fpp0_001_1000x1000s --algorithm huntress --fpp_override 0.01
poetry run fpp0_001_1000x1000s --algorithm scistreep --fpp_override 0.01
poetry run fpp0_001_1000x1000s --algorithm scite --fpp_override 0.01

echo ""
echo "1000x1000s robustness to FPP, true value is 0.01"
# echo ""
# echo "plug in true"
# poetry run fpp0_01_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_01_1000x1000s --algorithm huntress
# poetry run fpp0_01_1000x1000s --algorithm scistreep
# poetry run fpp0_01_1000x1000s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.001"
poetry run fpp0_01_1000x1000s --algorithm sempervirens_rs --fpp_override 0.001
poetry run fpp0_01_1000x1000s --algorithm huntress --fpp_override 0.001
poetry run fpp0_01_1000x1000s --algorithm scistreep --fpp_override 0.001
poetry run fpp0_01_1000x1000s --algorithm scite --fpp_override 0.001
echo ""
echo "plug in 10x -> 0.1"
poetry run fpp0_01_1000x1000s --algorithm sempervirens_rs --fpp_override 0.1
poetry run fpp0_01_1000x1000s --algorithm huntress --fpp_override 0.1
poetry run fpp0_01_1000x1000s --algorithm scistreep --fpp_override 0.1
poetry run fpp0_01_1000x1000s --algorithm scite --fpp_override 0.1

echo ""
echo "FNP"

echo ""
echo "1000x1000s robustness to FNP, true value is 0.05"
# echo ""
# echo "plug in true"
# poetry run fpp0_001_fnp0_05_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_01_fnp0_05_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_001_fnp0_05_1000x1000s --algorithm huntress
# poetry run fpp0_01_fnp0_05_1000x1000s --algorithm huntress
# poetry run fpp0_001_fnp0_05_1000x1000s --algorithm scistreep
# poetry run fpp0_01_fnp0_05_1000x1000s --algorithm scistreep
# poetry run fpp0_001_fnp0_05_1000x1000s --algorithm scite
# poetry run fpp0_01_fnp0_05_1000x1000s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.005"
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm sempervirens_rs --fnp_override 0.005
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm sempervirens_rs --fnp_override 0.005
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm huntress --fnp_override 0.005
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm huntress --fnp_override 0.005
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm scistreep --fnp_override 0.005
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm scistreep --fnp_override 0.005
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm scite --fnp_override 0.005
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm scite --fnp_override 0.005
echo ""
echo "plug in 10x -> 0.5"
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm sempervirens_rs --fnp_override 0.5
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm sempervirens_rs --fnp_override 0.5
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm huntress --fnp_override 0.5
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm huntress --fnp_override 0.5
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm scistreep --fnp_override 0.5
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm scistreep --fnp_override 0.5
poetry run fpp0_001_fnp0_05_1000x1000s --algorithm scite --fnp_override 0.5
poetry run fpp0_01_fnp0_05_1000x1000s --algorithm scite --fnp_override 0.5

echo ""
echo "1000x1000s robustness to FNP, true value is 0.2"
# echo ""
# echo "plug in true"
# poetry run fpp0_001_fnp0_2_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_01_fnp0_2_1000x1000s --algorithm sempervirens_rs
# poetry run fpp0_001_fnp0_2_1000x1000s --algorithm huntress
# poetry run fpp0_01_fnp0_2_1000x1000s --algorithm huntress
# poetry run fpp0_001_fnp0_2_1000x1000s --algorithm scistreep
# poetry run fpp0_01_fnp0_2_1000x1000s --algorithm scistreep
# poetry run fpp0_001_fnp0_2_1000x1000s --algorithm scite
# poetry run fpp0_01_fnp0_2_1000x1000s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.02"
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm sempervirens_rs --fnp_override 0.02
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm sempervirens_rs --fnp_override 0.02
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm huntress --fnp_override 0.02
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm huntress --fnp_override 0.02
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm scistreep --fnp_override 0.02
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm scistreep --fnp_override 0.02
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm scite --fnp_override 0.02
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm scite --fnp_override 0.02
echo ""
echo "plug in 2x -> 0.4"
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm sempervirens_rs --fnp_override 0.4
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm sempervirens_rs --fnp_override 0.4
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm huntress --fnp_override 0.4
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm huntress --fnp_override 0.4
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm scistreep --fnp_override 0.4
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm scistreep --fnp_override 0.4
poetry run fpp0_001_fnp0_2_1000x1000s --algorithm scite --fnp_override 0.4
poetry run fpp0_01_fnp0_2_1000x1000s --algorithm scite --fnp_override 0.4