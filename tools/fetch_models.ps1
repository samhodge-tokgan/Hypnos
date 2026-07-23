# Download the MEMatte ONNX models into the plugin bundle's Contents\Resources.
#
# Release installers are shipped MODEL-LESS to stay under GitHub's per-asset size
# limit. Run this once after installing:
#
#   powershell -ExecutionPolicy Bypass -File Contents\fetch_models.ps1
#   powershell -File tools\fetch_models.ps1 -Dest C:\models    # dev build
#
# You can skip this entirely by exporting the models yourself
# (tools\export_mematte.py) and pointing $env:MEMATTE_MODEL_DIR at them, or by
# setting the plugin's Backbone/Decoder file parameters.
param(
    [string]$Dest = "",
    [string]$BaseUrl = $(if ($env:MEMATTE_MODEL_BASE_URL) { $env:MEMATTE_MODEL_BASE_URL }
                         else { "https://github.com/samhodge-tokgan/Hypnos/releases/download/models-v1" }),
    [string[]]$Variants = @("s", "b")
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrEmpty($Dest)) {
    # Default to Resources\ next to this script (the in-bundle case).
    $Dest = Join-Path $PSScriptRoot "Resources"
}
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# SHA-256 of each published asset. Filled in when the models-v1 release is cut;
# until then the checksum step is skipped with a loud warning rather than
# silently trusting whatever the URL returns.
$Checksums = @{
    "mematte_s_backbone.onnx" = ""
    "mematte_s_decoder.onnx"  = ""
    "mematte_b_backbone.onnx" = ""
    "mematte_b_decoder.onnx"  = ""
}

foreach ($v in $Variants) {
    foreach ($part in @("backbone", "decoder")) {
        $name = "mematte_${v}_${part}.onnx"
        $out = Join-Path $Dest $name
        if (Test-Path $out) {
            Write-Host "$name already present, skipping"
            continue
        }
        Write-Host "fetching $name"
        $tmp = "$out.part"
        Invoke-WebRequest -Uri "$BaseUrl/$name" -OutFile $tmp -UseBasicParsing

        $want = $Checksums[$name]
        if ([string]::IsNullOrEmpty($want)) {
            Write-Warning "no pinned checksum for $name - NOT verified"
        } else {
            $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
            if ($got -ne $want.ToLower()) {
                Remove-Item $tmp -Force
                throw "CHECKSUM MISMATCH for ${name}: expected $want, got $got"
            }
            Write-Host "  checksum ok"
        }
        Move-Item -Force $tmp $out
    }
}

Write-Host ""
Write-Host "models are in: $Dest"
Get-ChildItem $Dest
