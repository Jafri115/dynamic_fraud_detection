# Define your root path
$projectRoot = "D:\git_projects\dynamic_fraud_detection\"
$outputFile = Join-Path $projectRoot "project_structure.txt"

# Define ignored folders (case-insensitive)
$ignoredDirs = @(
    ".venv", "__pycache__", "logs", "catboost_info", "*.egg-info", "node_modules", "dist", "build", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints", "coverage_html_report", "coverage", "htmlcov", ".tox", ".eggs",
    "pip-wheel-metadata", "pip-egg-info", "site-packages", ".git", ".github", ".idea", ".vscode", "docs", "tests", "test_results", "reports", "scripts", "examples", "notebooks", "assets"
)

# Define ignored file extensions
$ignoredExtensions = @(
    ".pyc", ".tmp", ".tsv", ".json"
)

# Function to recursively print folder tree
function Print-Tree {
    param (
        [string]$path,
        [int]$indent = 0
    )

    $prefix = ("|   " * $indent)
    $folderName = Split-Path $path -Leaf
    Add-Content $outputFile "$prefix+-- $folderName"

    # Get child directories excluding ignored ones
    $dirs = Get-ChildItem -Path $path -Directory -Force |
        Where-Object { $ignoredDirs -notcontains $_.Name -and ($_ -notmatch "\.egg-info$") }

    # Get child files excluding unimportant extensions
    $files = Get-ChildItem -Path $path -File -Force |
        Where-Object { $ignoredExtensions -notcontains $_.Extension }

    foreach ($file in $files) {
        Add-Content $outputFile "$prefix|   +-- $($file.Name)"
    }

    foreach ($dir in $dirs) {
        Print-Tree -path $dir.FullName -indent ($indent + 1)
    }
}

# Start fresh
if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

# Write the root folder
Add-Content $outputFile "$(Split-Path $projectRoot -Leaf)/"

# Generate the tree
Print-Tree -path $projectRoot -indent 0

Write-Host "`n✅ Project structure saved to:`n$outputFile"
