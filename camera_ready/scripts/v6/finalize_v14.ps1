$s = "C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin\1282218c-8e1c-4ad5-a29d-8076480f7dbc\scratchpad\v6"
Push-Location "$s\fig7"; python -X utf8 make_fig4.py | Select-Object -Last 1; Pop-Location
Copy-Item "$s\fig7\newfigs\fig4.png" "$s\unpacked\word\media\image4.png" -Force
python -X utf8 -I "$s\set_fig_aspect.py" image4.png | Select-Object -Last 1
& "$s\build_v11.ps1" -tag v14
python -X utf8 -I "$s\page_gaps.py" "$s\ICAIN2026_456_NeuroClick_CRC_v14.pdf" | Select-Object -Last 1
python -I "$s\inspect_pdf.py" "$s\ICAIN2026_456_NeuroClick_CRC_v14.pdf" | Select-String "NOT embedded" | Measure-Object -Line | ForEach-Object { "unembedded fonts: " + $_.Lines }
python -X utf8 -I "$s\check_numbers.py" | Select-Object -First 1
# deliverables
Copy-Item "$s\ICAIN2026_456_NeuroClick_CRC_v14.docx", "$s\ICAIN2026_456_NeuroClick_CRC_v14.pdf" "D:\PRIYA\ICAIN" -Force
foreach ($dst in @("D:\PRIYA\ICAIN\camera_ready_build", "D:\PRIYA\ICAIN\neuroclick_repo\camera_ready")) {
    foreach ($i in 1..7) { Copy-Item "$s\fig7\newfigs\fig$i.png" "$dst\figures\fig$i.png" -Force }
    New-Item -ItemType Directory -Force "$dst\scripts\v6\figures_8pt" | Out-Null
    Copy-Item "$s\fig7\make_fig1.py", "$s\fig7\make_fig2_v14.py", "$s\fig7\make_fig3.py", "$s\fig7\make_fig4.py", "$s\fig7\make_fig5.py", "$s\fig7\make_fig6.py", "$s\fig7\make_fig7.py", "$s\fig7\figlib.py" "$dst\scripts\v6\figures_8pt" -Force
    Copy-Item "$s\set_fig_aspect.py", "$s\unkeep_table7.py", "$s\page_gaps.py", "$s\finalize_v14.ps1" "$dst\scripts\v6" -Force
}
Get-ChildItem "D:\PRIYA\ICAIN" -Filter "*v14*" | ForEach-Object { $_.Name + "  " + $_.Length + "  " + (Get-FileHash $_.FullName -Algorithm SHA256).Hash.Substring(0, 12) }
$git = "C:\Users\Admin\AppData\Local\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe"
Set-Location "D:\PRIYA\ICAIN\neuroclick_repo"
& $git -c core.safecrlf=false add -A 2>$null
& $git -c user.name="Priyadharshini D" -c user.email="priyadharshini.2024b@vitstudent.ac.in" -c core.safecrlf=false commit -q -m "Camera-ready v14: all seven figures regenerated from repository data with 8 pt minimum lettering" 2>$null
& $git push origin main 2>$null | Out-String
& $git fetch origin 2>$null
"HEAD " + (& $git rev-parse --short HEAD) + "  origin/main " + (& $git rev-parse --short origin/main)
