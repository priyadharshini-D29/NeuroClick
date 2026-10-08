param([string]$docx, [string]$pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $true)
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $false, $true, 0, $true, $true, $false)
    "pages: $($doc.ComputeStatistics(2))"
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
"exported $pdf"
