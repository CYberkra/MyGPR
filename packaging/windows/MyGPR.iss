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
#ifndef AppFileVersion
  #define AppFileVersion AppVersion + ".0"
#endif
VersionInfoVersion={#AppFileVersion}
VersionInfoProductVersion={#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
AppMutex=Local\MyGPR.Application
CloseApplications=no
RestartApplications=no
#ifdef MyGPRSigned
SignTool=MyGPRSigner
SignedUninstaller=yes
#endif
LicenseFile={#DistDir}\_internal\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#DistDir}\mygpr-managed-files.txt"; Flags: dontcopy
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MyGPR"; Filename: "{app}\MyGPR.exe"
Name: "{autodesktop}\MyGPR"; Filename: "{app}\MyGPR.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MyGPR.exe"; Description: "Launch MyGPR"; Flags: nowait postinstall skipifsilent; Check: not IsUpdate
Filename: "{app}\MyGPR.exe"; Flags: nowait runasoriginaluser; Check: IsUpdate

[Code]
function FileAttributes(Name: String): LongWord;
  external 'GetFileAttributesW@kernel32.dll stdcall';

function IsUpdate: Boolean;
begin
  Result := ExpandConstant('{param:MYGPRUPDATE|0}') = '1';
end;

function SafeManagedPath(RelativeName: String): Boolean;
var
  Parent: String;
begin
  Result := False;
  if (Pos('..', RelativeName) > 0) or (Pos(':', RelativeName) > 0) or
     (Pos('/', RelativeName) > 0) or (Copy(RelativeName, 1, 10) <> '_internal\') then Exit;
  Parent := ExpandConstant('{app}') + '\' + RelativeName;
  { Never traverse a junction or a symlink, including the installed root. }
  while Length(Parent) >= Length(ExpandConstant('{app}')) do begin
    if (FileAttributes(Parent) and $400) <> 0 then Exit;
    Parent := ExtractFileDir(Parent);
  end;
  Result := True;
end;

procedure RemoveObsoleteManagedFiles;
var
  OldFiles, NewFiles: TArrayOfString;
  I, J, Separator: Integer;
  Entry, RelativeName, OldHash, FileName: String;
  StillShipped: Boolean;
  NewNames: TStringList;
begin
  if not LoadStringsFromFile(ExpandConstant('{app}\mygpr-managed-files.txt'), OldFiles) then Exit;
  ExtractTemporaryFile('mygpr-managed-files.txt');
  if not LoadStringsFromFile(ExpandConstant('{tmp}\mygpr-managed-files.txt'), NewFiles) then Exit;
  NewNames := TStringList.Create;
  try
  NewNames.Sorted := True;
  NewNames.CaseSensitive := False;
  for J := 0 to GetArrayLength(NewFiles) - 1 do
    NewNames.Add(Copy(NewFiles[J], 66, Length(NewFiles[J])));
  for I := 0 to GetArrayLength(OldFiles) - 1 do begin
    Entry := OldFiles[I];
    Separator := Pos('|', Entry);
    if Separator <> 65 then Continue;
    OldHash := Copy(Entry, 1, 64);
    RelativeName := Copy(Entry, 66, Length(Entry));
    StringChangeEx(RelativeName, '/', '\', True);
    if not SafeManagedPath(RelativeName) then Continue;
    StillShipped := NewNames.IndexOf(Copy(Entry, 66, Length(Entry))) >= 0;
    FileName := ExpandConstant('{app}') + '\' + RelativeName;
    if not StillShipped and FileExists(FileName) then begin
      try
        if CompareText(GetSHA256OfFile(FileName), OldHash) = 0 then
          if not DeleteFile(FileName) then
            Log('Preserved obsolete file that could not be deleted: ' + RelativeName);
      except
        Log('Preserved obsolete file that could not be verified: ' + RelativeName);
      end;
    end;
  end;
  finally
    NewNames.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then RemoveObsoleteManagedFiles;
end;

{ User projects, acquisition files and ~/MyGPR/config are outside
  the installer file table. Uninstall removes installed files, not user data. }
