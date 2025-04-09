# PowerShell script to convert remaining IMG files to TIFF format using a loop
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Starting image conversion process..."
Write-Host "========================================"

# Define the base path
$basePath = "\\10.68.54.238\wcx\哈萨克斯坦冰凌\乌拉尔河\乌拉尔河高分影像"

# List of base filenames (without .img extension) for remaining files
$fileBases = @(
    "GF1_PMS2_E51.6_N49.8_20231213_L1A13202129001-MSS2_fuse",
    "GF1_PMS2_E51.6_N49.8_20240414_L1A13364641001-MSS2_fuse",
    "GF1_PMS2_E52.2_N51.2_20231213_L1A13202124001-MSS2_fuse",
    "GF1B_PMS_E51.9_N49.6_20231208_L1A1228397064-MUX_fuse",
    "GF1B_PMS_E51.9_N49.6_20240228_L1A1228454567-MUX_fuse",
    "GF1B_PMS_E52.6_N51.2_20231208_L1A1228397068-MUX_fuse",
    "GF1C_PMS_E51.9_N49.6_20240329_L1A1022230270-MUX_fuse",
    "GF1C_PMS_E52.6_N51.2_20231127_L1A1022157800-MUX_fuse",
    "GF1C_PMS_E52.7_N51.6_20231127_L1A1022157804-MUX_fuse",
    "GF6_PMS_E52.0_N49.8_20240129_L1A1420411591-MUX_fuse"
)

# Loop through each base filename
foreach ($baseName in $fileBases) {
    $inputFile = Join-Path -Path $basePath -ChildPath ($baseName + ".img")
    $outputDir = Join-Path -Path $basePath -ChildPath ($baseName + "_output\") # Ensure trailing slash for directory

    Write-Host "Processing: $inputFile"

    # Execute the python script
    python main.py img-to-tiff -i "$inputFile" -o "$outputDir"

    # Check for errors
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Error processing $inputFile. Exit code: $LASTEXITCODE"
    } else {
        # Assuming the python script prints its own success message upon completion
        Write-Host "Finished processing call for $inputFile." -ForegroundColor Cyan
    }
    Write-Host "---"
}

Write-Host "========================================"
Write-Host "All image conversion calls finished."