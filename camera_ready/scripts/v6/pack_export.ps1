param([string]$base, [string]$tag)
# 1. zip unpacked -> docx
$docx = Join-Path $base "$tag.docx"
if (Test-Path $docx) { Remove-Item $docx -Force }
python -I (Join-Path $base "pack.py") (Join-Path $base "unpacked") $docx
# 2. Word 2010 COM export; UseISO19005_1 = $true (PDF/A) forces every font to be embedded
$pdf = Join-Path $base "$tag.pdf"
if (Test-Path $pdf) { Remove-Item $pdf -Force }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $true)
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $false, $true, 0, $true, $true, $true)
    "pages: $($doc.ComputeStatistics(2))"
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
"exported $pdf"
