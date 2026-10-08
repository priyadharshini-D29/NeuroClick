param([string]$tag = "v11", [string]$edit = "")
$s = "C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin\1282218c-8e1c-4ad5-a29d-8076480f7dbc\scratchpad\v6"
$name = "ICAIN2026_456_NeuroClick_CRC_$tag"
if ($edit -ne "") {
    python -I "$s\$edit"
    if (-not $?) { throw "edit failed" }
}
python -I "$s\pack.py" "$s\unpacked" "$s\$name.docx"
$docx = "$s\$name.docx"
$pdf = "$s\${name}_raw.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $true)
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $false, $true, 0, $true, $true, $true)
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
python -I "$s\fix_images.py" $pdf "$s\$name.pdf" "$s\unpacked" | Select-Object -Last 1
Remove-Item $pdf -Force
python -I "$s\finish_any.py" "$s\$name.pdf"
python -I "$s\inspect_pdf.py" "$s\$name.pdf" | Select-String "NOT embedded" | Measure-Object -Line | ForEach-Object { "unembedded fonts: " + $_.Lines }
