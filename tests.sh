#! /usr/bin/sh

mkdir -p /tmp/fp_0_001
mkdir -p /tmp/fp_0_01

echo ""
echo "FPP"

echo ""
echo "300x300s robustness to FPP, true value is 0.001"
echo ""
echo "plug in true"
poetry run fpr0_001_300x300s --algorithm sempervirens_rs
poetry run fpr0_001_300x300s --algorithm huntress
poetry run fpr0_001_300x300s --algorithm scistreep
poetry run fpr0_001_300x300s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.0001"
poetry run fpr0_001_300x300s --algorithm sempervirens_rs --fpp_override 0.0001
poetry run fpr0_001_300x300s --algorithm huntress --fpp_override 0.0001
poetry run fpr0_001_300x300s --algorithm scistreep --fpp_override 0.0001
poetry run fpr0_001_300x300s --algorithm scite --fpp_override 0.0001
echo ""
echo "plug in 10x -> 0.01"
poetry run fpr0_001_300x300s --algorithm sempervirens_rs --fpp_override 0.01
poetry run fpr0_001_300x300s --algorithm huntress --fpp_override 0.01
poetry run fpr0_001_300x300s --algorithm scistreep --fpp_override 0.01
poetry run fpr0_001_300x300s --algorithm scite --fpp_override 0.01

echo ""
echo "300x300s robustness to FPP, true value is 0.01"
echo ""
echo "plug in true"
poetry run fpr0_01_300x300s --algorithm sempervirens_rs
poetry run fpr0_01_300x300s --algorithm huntress
poetry run fpr0_01_300x300s --algorithm scistreep
poetry run fpr0_01_300x300s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.001"
poetry run fpr0_01_300x300s --algorithm sempervirens_rs --fpp_override 0.001
poetry run fpr0_01_300x300s --algorithm huntress --fpp_override 0.001
poetry run fpr0_01_300x300s --algorithm scistreep --fpp_override 0.001
poetry run fpr0_01_300x300s --algorithm scite --fpp_override 0.001
echo ""
echo "plug in 10x -> 0.1"
poetry run fpr0_01_300x300s --algorithm sempervirens_rs --fpp_override 0.1
poetry run fpr0_01_300x300s --algorithm huntress --fpp_override 0.1
poetry run fpr0_01_300x300s --algorithm scistreep --fpp_override 0.1
poetry run fpr0_01_300x300s --algorithm scite --fpp_override 0.1

echo ""
echo "FNP"

echo ""
echo "300x300s robustness to FNP, true value is 0.05"
echo ""
echo "plug in true"
poetry run fpr0_001_fnr0_05_300x300s --algorithm sempervirens_rs
poetry run fpr0_01_fnr0_05_300x300s --algorithm sempervirens_rs
poetry run fpr0_001_fnr0_05_300x300s --algorithm huntress
poetry run fpr0_01_fnr0_05_300x300s --algorithm huntress
poetry run fpr0_001_fnr0_05_300x300s --algorithm scistreep
poetry run fpr0_01_fnr0_05_300x300s --algorithm scistreep
poetry run fpr0_001_fnr0_05_300x300s --algorithm scite
poetry run fpr0_01_fnr0_05_300x300s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.005"
poetry run fpr0_001_fnr0_05_300x300s --algorithm sempervirens_rs --fnp_override 0.005
poetry run fpr0_01_fnr0_05_300x300s --algorithm sempervirens_rs --fnp_override 0.005
poetry run fpr0_001_fnr0_05_300x300s --algorithm huntress --fnp_override 0.005
poetry run fpr0_01_fnr0_05_300x300s --algorithm huntress --fnp_override 0.005
poetry run fpr0_001_fnr0_05_300x300s --algorithm scistreep --fnp_override 0.005
poetry run fpr0_01_fnr0_05_300x300s --algorithm scistreep --fnp_override 0.005
poetry run fpr0_001_fnr0_05_300x300s --algorithm scite --fnp_override 0.005
poetry run fpr0_01_fnr0_05_300x300s --algorithm scite --fnp_override 0.005
echo ""
echo "plug in 10x -> 0.5"
poetry run fpr0_001_fnr0_05_300x300s --algorithm sempervirens_rs --fnp_override 0.5
poetry run fpr0_01_fnr0_05_300x300s --algorithm sempervirens_rs --fnp_override 0.5
poetry run fpr0_001_fnr0_05_300x300s --algorithm huntress --fnp_override 0.5
poetry run fpr0_01_fnr0_05_300x300s --algorithm huntress --fnp_override 0.5
poetry run fpr0_001_fnr0_05_300x300s --algorithm scistreep --fnp_override 0.5
poetry run fpr0_01_fnr0_05_300x300s --algorithm scistreep --fnp_override 0.5
poetry run fpr0_001_fnr0_05_300x300s --algorithm scite --fnp_override 0.5
poetry run fpr0_01_fnr0_05_300x300s --algorithm scite --fnp_override 0.5

echo ""
echo "300x300s robustness to FNP, true value is 0.2"
echo ""
echo "plug in true"
poetry run fpr0_001_fnr0_2_300x300s --algorithm sempervirens_rs
poetry run fpr0_01_fnr0_2_300x300s --algorithm sempervirens_rs
poetry run fpr0_001_fnr0_2_300x300s --algorithm huntress
poetry run fpr0_01_fnr0_2_300x300s --algorithm huntress
poetry run fpr0_001_fnr0_2_300x300s --algorithm scistreep
poetry run fpr0_01_fnr0_2_300x300s --algorithm scistreep
poetry run fpr0_001_fnr0_2_300x300s --algorithm scite
poetry run fpr0_01_fnr0_2_300x300s --algorithm scite
echo ""
echo "plug in 0.1x -> 0.02"
poetry run fpr0_001_fnr0_2_300x300s --algorithm sempervirens_rs --fnp_override 0.02
poetry run fpr0_01_fnr0_2_300x300s --algorithm sempervirens_rs --fnp_override 0.02
poetry run fpr0_001_fnr0_2_300x300s --algorithm huntress --fnp_override 0.02
poetry run fpr0_01_fnr0_2_300x300s --algorithm huntress --fnp_override 0.02
poetry run fpr0_001_fnr0_2_300x300s --algorithm scistreep --fnp_override 0.02
poetry run fpr0_01_fnr0_2_300x300s --algorithm scistreep --fnp_override 0.02
poetry run fpr0_001_fnr0_2_300x300s --algorithm scite --fnp_override 0.02
poetry run fpr0_01_fnr0_2_300x300s --algorithm scite --fnp_override 0.02
echo ""
echo "plug in 2x -> 0.4"
poetry run fpr0_001_fnr0_2_300x300s --algorithm sempervirens_rs --fnp_override 0.4
poetry run fpr0_01_fnr0_2_300x300s --algorithm sempervirens_rs --fnp_override 0.4
poetry run fpr0_001_fnr0_2_300x300s --algorithm huntress --fnp_override 0.4
poetry run fpr0_01_fnr0_2_300x300s --algorithm huntress --fnp_override 0.4
poetry run fpr0_001_fnr0_2_300x300s --algorithm scistreep --fnp_override 0.4
poetry run fpr0_01_fnr0_2_300x300s --algorithm scistreep --fnp_override 0.4
poetry run fpr0_001_fnr0_2_300x300s --algorithm scite --fnp_override 0.4
poetry run fpr0_01_fnr0_2_300x300s --algorithm scite --fnp_override 0.4