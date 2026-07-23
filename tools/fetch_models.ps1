# Put the MEMatte ONNX models into the plugin bundle's Contents\Resources.
#
# THIS PROJECT DOES NOT REDISTRIBUTE MODEL WEIGHTS, and there is no default
# download URL. That is deliberate, not an oversight:
#
#   * MEMatte licenses its CODE as MIT ("The code is released under the MIT
#     License"); it says nothing about the weights and ships no LICENSE file.
#   * The published checkpoints (MEMatte_ViT{S,B}_DIM.pth) are trained on the
#     Adobe Deep Image Matting / Composition-1k dataset, and fine-tuned from
#     ViTMatte's Composition-1k checkpoints. Adobe's dataset agreement restricts
#     models trained on it to NON-COMMERCIAL use and distribution.
#
# So exporting the models is something you do yourself, having accepted the
# upstream terms. It takes about a minute per variant:
#
#     git clone https://github.com/linyiheng123/MEMatte
#     pip install -r tools\requirements.txt
#     pip install 'git+https://github.com/facebookresearch/detectron2.git'
#     python tools\export_mematte.py --mematte-repo ..\MEMatte `
#         --checkpoint ..\MEMatte\checkpoints\MEMatte_ViTS_DIM.pth `
#         --variant s --out-dir build\models
#
# Then either copy the .onnx files into this bundle's Contents\Resources, point
# $env:MEMATTE_MODEL_DIR at them, or set the plugin's Backbone/Decoder file params.
#
# If your facility mirrors the exported models internally, point this script at
# that mirror and it will fetch and (optionally) checksum them:
#
#     powershell -File fetch_models.ps1 -BaseUrl https://internal.example/mematte
param(
    [string]$Dest = "",
    [string]$BaseUrl = $(if ($env:MEMATTE_MODEL_BASE_URL) { $env:MEMATTE_MODEL_BASE_URL } else { "" }),
    [string]$Sha256File = "",
    [string[]]$Variants = @("s", "b")
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrEmpty($BaseUrl)) {
    Get-Content $PSCommandPath | Select-Object -First 29 | ForEach-Object { $_ -replace '^# ?', '' }
    Write-Host ""
    Write-Error ("No -BaseUrl given, and this project publishes no model weights. " +
                 "Export them yourself with tools\export_mematte.py (see above).")
    exit 1
}

if ([string]::IsNullOrEmpty($Dest)) {
    # Default to Resources\ next to this script (the in-bundle case).
    $Dest = Join-Path $PSScriptRoot "Resources"
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$Checksums = @{}
if (-not [string]::IsNullOrEmpty($Sha256File)) {
    Get-Content $Sha256File | ForEach-Object {
        $parts = $_ -split '\s+' | Where-Object { $_ }
        if ($parts.Count -ge 2) { $Checksums[($parts[1] -replace '^\*', '')] = $parts[0] }
    }
}

foreach ($v in $Variants) {
    foreach ($part in @("backbone", "decoder")) {
        $name = "mematte_${v}_${part}.onnx"
        $out = Join-Path $Dest $name
        if (Test-Path $out) {
            Write-Host "$name already present, skipping"
            continue
        }
        Write-Host "fetching $name from $BaseUrl"
        $tmp = "$out.part"
        Invoke-WebRequest -Uri "$BaseUrl/$name" -OutFile $tmp -UseBasicParsing

        if ($Checksums.ContainsKey($name)) {
            $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
            if ($got -ne $Checksums[$name].ToLower()) {
                Remove-Item $tmp -Force
                throw "CHECKSUM MISMATCH for ${name}: expected $($Checksums[$name]), got $got"
            }
            Write-Host "  checksum ok"
        } else {
            Write-Warning "no checksum listed for $name - NOT verified"
        }
        Move-Item -Force $tmp $out
    }
}

Write-Host ""
Write-Host "models are in: $Dest"
Get-ChildItem $Dest
