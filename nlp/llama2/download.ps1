#Requires -Version 5.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

param(
    [string]$PresignedUrl = (Read-Host -Prompt "Enter the URL from email"),
    [string]$ModelSizeInput = (Read-Host -Prompt "Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all")
)

$TargetFolder = "." # where all files should end up

# Create target folder if it doesn't exist
if (-not (Test-Path -Path $TargetFolder -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $TargetFolder | Out-Null
}

$ModelSize = $ModelSizeInput
if ([string]::IsNullOrWhiteSpace($ModelSize)) {
    $ModelSize = "7B,13B,70B,7B-chat,13B-chat,70B-chat"
}

$Models = $ModelSize -split ','

Write-Host "Downloading LICENSE and Acceptable Usage Policy"
try {
    Invoke-WebRequest -Uri $PresignedUrl.Replace('*', 'LICENSE') -OutFile (Join-Path $TargetFolder "LICENSE") -ErrorAction Stop
    Invoke-WebRequest -Uri $PresignedUrl.Replace('*', 'USE_POLICY.md') -OutFile (Join-Path $TargetFolder "USE_POLICY.md") -ErrorAction Stop
} catch {
    Write-Error "Failed to download LICENSE or USE_POLICY.md: $($_.Exception.Message)"
    # Decide if script should exit or continue
    # exit 1
}

Write-Host "Downloading tokenizer"
try {
    $TokenizerModelPath = Join-Path $TargetFolder "tokenizer.model"
    $TokenizerChecklistPath = Join-Path $TargetFolder "tokenizer_checklist.chk"
    Invoke-WebRequest -Uri $PresignedUrl.Replace('*', 'tokenizer.model') -OutFile $TokenizerModelPath -ErrorAction Stop
    Invoke-WebRequest -Uri $PresignedUrl.Replace('*', 'tokenizer_checklist.chk') -OutFile $TokenizerChecklistPath -ErrorAction Stop

    # Checksum verification for tokenizer
    Write-Host "Checking tokenizer checksums..."
    Push-Location $TargetFolder
    $checksums = Get-Content $TokenizerChecklistPath
    $allOk = $true
    foreach ($line in $checksums) {
        $parts = $line.Trim() -split '\s+', 2
        if ($parts.Length -ne 2) { continue }
        $expectedHash = $parts[0]
        $fileName = $parts[1].TrimStart('*') # Remove leading '*' if present
        $filePath = Join-Path "." $fileName # Relative to current dir (TargetFolder)
        if (Test-Path $filePath) {
            $actualHash = (Get-FileHash -Algorithm MD5 -Path $filePath).Hash.ToLower()
            if ($actualHash -ne $expectedHash.ToLower()) {
                Write-Warning "Checksum mismatch for ${fileName}: Expected ${expectedHash}, Got ${actualHash}"
                $allOk = $false
            } else {
                Write-Host "Checksum OK for $fileName"
            }
        } else {
            Write-Warning "File not found for checksum verification: ${fileName}"
            $allOk = $false
        }
    }
    Pop-Location
    if (-not $allOk) {
        Write-Error "Tokenizer checksum verification failed."
        # exit 1 # Optionally exit if verification fails
    } else {
        Write-Host "Tokenizer checksums verified successfully."
    }

} catch {
    Write-Error "Failed to download tokenizer files: $($_.Exception.Message)"
    # exit 1
}


foreach ($m in $Models) {
    $m = $m.Trim()
    $Shard = 0
    $ModelPathName = ""

    switch ($m) {
        "7B"       { $Shard = 0; $ModelPathName = "llama-2-7b" }
        "7B-chat"  { $Shard = 0; $ModelPathName = "llama-2-7b-chat" }
        "13B"      { $Shard = 1; $ModelPathName = "llama-2-13b" }
        "13B-chat" { $Shard = 1; $ModelPathName = "llama-2-13b-chat" }
        "70B"      { $Shard = 7; $ModelPathName = "llama-2-70b" }
        "70B-chat" { $Shard = 7; $ModelPathName = "llama-2-70b-chat" }
        default    { Write-Warning "Unknown model size: $m"; continue }
    }

    Write-Host "Downloading $ModelPathName"
    $ModelDir = Join-Path $TargetFolder $ModelPathName
    if (-not (Test-Path -Path $ModelDir -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    }

    # Download model shards
    # Note: Invoke-WebRequest doesn't support resuming downloads like wget --continue.
    # Consider Start-BitsTransfer for large files if resuming is critical.
    for ($s = 0; $s -le $Shard; $s++) {
        $ShardFileName = "consolidated.${s}.pth"
        $ShardUrl = $PresignedUrl.Replace('*', "$ModelPathName/$ShardFileName")
        $ShardOutFile = Join-Path $ModelDir $ShardFileName
        Write-Host "Downloading $ShardFileName..."
        try {
            # Add -UseBasicParsing for potentially better compatibility/performance
            # Add retry logic manually if needed, as Invoke-WebRequest doesn't have built-in retries like wget's -t option.
            Invoke-WebRequest -Uri $ShardUrl -OutFile $ShardOutFile -UseBasicParsing -ErrorAction Stop
        } catch {
            Write-Error "Failed to download ${ShardFileName} from ${ShardUrl} : $($_.Exception.Message)"
            # Optionally break or continue to next model
            # break # Stop downloading this model
            # continue # Continue with the next file/model (might leave model incomplete)
        }
    }

    # Download params.json and checklist.chk
    try {
        $ParamsJsonUrl = $PresignedUrl.Replace('*', "$ModelPathName/params.json")
        $ParamsJsonOutFile = Join-Path $ModelDir "params.json"
        Invoke-WebRequest -Uri $ParamsJsonUrl -OutFile $ParamsJsonOutFile -ErrorAction Stop

        $ChecklistUrl = $PresignedUrl.Replace('*', "$ModelPathName/checklist.chk")
        $ChecklistOutFile = Join-Path $ModelDir "checklist.chk"
        Invoke-WebRequest -Uri $ChecklistUrl -OutFile $ChecklistOutFile -ErrorAction Stop

        # Checksum verification for the model
        Write-Host "Checking checksums for $ModelPathName"
        Push-Location $ModelDir
        $checksums = Get-Content $ChecklistOutFile
        $allOk = $true
        foreach ($line in $checksums) {
            $parts = $line.Trim() -split '\s+', 2
            if ($parts.Length -ne 2) { continue }
            $expectedHash = $parts[0]
            $fileName = $parts[1].TrimStart('*') # Remove leading '*' if present
            $filePath = Join-Path "." $fileName # Relative to current dir (ModelDir)
            if (Test-Path $filePath) {
                Write-Host "Calculating hash for $fileName..."
                $actualHash = (Get-FileHash -Algorithm MD5 -Path $filePath).Hash.ToLower()
                if ($actualHash -ne $expectedHash.ToLower()) {
                    Write-Warning "Checksum mismatch for ${fileName}: Expected ${expectedHash}, Got ${actualHash}"
                    $allOk = $false
                } else {
                     Write-Host "Checksum OK for $fileName"
                }
            } else {
                Write-Warning "File not found for checksum verification: ${fileName}"
                $allOk = $false
            }
        }
        Pop-Location
        if (-not $allOk) {
            Write-Error "Checksum verification failed for ${ModelPathName}."
            # Decide how to handle failure, e.g., continue or exit
        } else {
            Write-Host "Checksums verified successfully for $ModelPathName."
        }
    } catch {
        Write-Error "Failed to download params.json or checklist.chk for ${ModelPathName}: $($_.Exception.Message)"
        # Decide how to handle failure
    }
}

Write-Host "Download process finished." 