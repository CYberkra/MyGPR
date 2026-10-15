#ifndef AppVersion
  #error AppVersion must be supplied by the build script
#endif
#ifndef DistDir
  #error DistDir must be supplied by the build script
#endif
#ifndef OutputDir
  #error OutputDir must be supplied by the build script
#endif
#ifndef IconPath
  #error IconPath must be supplied by the build script
#endif

[Setup]
AppId={{3B6B778E-9BFA-4D07-B321-3C58D39EAA04}
AppName=MyGPR
AppVersion={#AppVersion}
AppVerName=MyGPR {#AppVersion}
AppPublisher=MyGPR Project
AppPublisherURL=https://github.com/CYberkra/MyGPR
AppSupportURL=https://github.com/CYberkra/MyGPR/issues
AppUpdatesURL=https://github.com/CYberkra/MyGPR/releases
DefaultDirName={localappdata}\Programs\MyGPR
DefaultGroupName=MyGPR
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=MyGPR-{#AppVersion}-Windows-x64-Setup
SetupIconFile={#IconPath}
UninstallDisplayIcon={app}\MyGPR.exe
VersionInfoVersion={#AppVersion}.0
VersionInfoProductVersion={#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
LicenseFile={#DistDir}\_internal\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MyGPR"; Filename: "{app}\MyGPR.exe"
Name: "{autodesktop}\MyGPR"; Filename: "{app}\MyGPR.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MyGPR.exe"; Description: "Launch MyGPR"; Flags: nowait postinstall skipifsilent

; User projects, acquisition files and ~/MyGPR/config are deliberately outside
; the installer file table. Uninstall removes installed files, not user data.
