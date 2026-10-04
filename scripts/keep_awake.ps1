# Hold the system + display awake while a long benchmark runs (Modern Standby needs both; see Phase 3 notes).
# Usage: powershell -File scripts/keep_awake.ps1 -Hours 14
param([double]$Hours = 14)
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000003") | Out-Null
Start-Sleep -Seconds ([int]($Hours * 3600))
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
