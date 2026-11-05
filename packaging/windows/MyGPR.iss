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
VersionInfoProductVersion={#AppFileVersion}
VersionInfoTextVersion={#AppVersion}
VersionInfoProductTextVersion={#AppVersion}
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
Source: "mygpr-0.11.0-managed-files.txt"; Flags: dontcopy
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

function InitializeSetup: Boolean;
var
  Attempts: Integer;
begin
  { WinSparkle launches Setup before requesting a normal application close. }
  if IsUpdate then begin
    Log('Waiting for MyGPR to exit before upgrade');
    Attempts := 0;
    while CheckForMutexes('Local\MyGPR.Application') and (Attempts < 300) do begin
      Sleep(100);
      Attempts := Attempts + 1;
    end;
  end;
  { If still open, the regular AppMutex prompt offers retry/cancel. }
  Result := True;
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

procedure RemoveEmptyManagedParents(FileName: String);
var
  Parent, InternalRoot: String;
begin
  Parent := ExtractFileDir(FileName);
  InternalRoot := ExpandConstant('{app}\_internal');
  while Length(Parent) > Length(InternalRoot) do begin
    if not RemoveDir(Parent) then Exit;
    Parent := ExtractFileDir(Parent);
  end;
end;

function LoadMigrationFiles(var OldFiles, NewFiles: TArrayOfString): Boolean;
begin
  if not LoadStringsFromFile(ExpandConstant('{app}\mygpr-managed-files.txt'), OldFiles) then begin
    { The published 0.11.0 installer predates package inventories. }
    ExtractTemporaryFile('mygpr-0.11.0-managed-files.txt');
    if not LoadStringsFromFile(ExpandConstant('{tmp}\mygpr-0.11.0-managed-files.txt'), OldFiles) then begin
      Result := False;
      Exit;
    end;
  end;
  ExtractTemporaryFile('mygpr-managed-files.txt');
  Result := LoadStringsFromFile(ExpandConstant('{tmp}\mygpr-managed-files.txt'), NewFiles);
end;

function RecordedDirectory(Name: String; Files: TArrayOfString): Boolean;
var
  I: Integer;
  Entry: String;
begin
  Result := False;
  for I := 0 to GetArrayLength(Files) - 1 do begin
    Entry := Copy(Files[I], 66, Length(Files[I]));
    StringChangeEx(Entry, '/', '\', True);
    if Pos(Lowercase(Name + '\'), Lowercase(Entry)) = 1 then begin
      Result := True;
      Exit;
    end;
  end;
end;

function MetadataIsUnmodified(Name: String; Files: TArrayOfString): Boolean;
var
  Found: TFindRec;
  I: Integer;
  Child, Entry, Expected: String;
begin
  Result := False;
  if not SafeManagedPath(Name) then Exit;
  if FindFirst(ExpandConstant('{app}\') + Name + '\*', Found) then begin
    try
      repeat
        if (Found.Name = '.') or (Found.Name = '..') then Continue;
        Child := Name + '\' + Found.Name;
        if (Found.Attributes and $10) <> 0 then begin
          if not RecordedDirectory(Child, Files) then Exit;
          if not MetadataIsUnmodified(Child, Files) then Exit;
        end else begin
          Expected := '';
          for I := 0 to GetArrayLength(Files) - 1 do begin
            Entry := Copy(Files[I], 66, Length(Files[I]));
            StringChangeEx(Entry, '/', '\', True);
            if CompareText(Child, Entry) = 0 then begin
              Expected := Copy(Files[I], 1, 64);
              Break;
            end;
          end;
          if (Expected = '') or not SafeManagedPath(Child) then Exit;
          try
            if CompareText(GetSHA256OfFile(ExpandConstant('{app}\') + Child), Expected) <> 0 then Exit;
          except
            Exit;
          end;
        end;
      until not FindNext(Found);
    finally
      FindClose(Found);
    end;
  end;
  Result := True;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  OldFiles, NewFiles: TArrayOfString;
  Found: TFindRec;
  Name: String;
begin
  Result := '';
  if not LoadMigrationFiles(OldFiles, NewFiles) then begin
    Result := 'Cannot read the package file inventory.';
    Exit;
  end;
  if FindFirst(ExpandConstant('{app}\_internal\mygpr-*.dist-info'), Found) then begin
    try
      repeat
        Name := '_internal\' + Found.Name;
        if ((Found.Attributes and $10) <> 0) and not RecordedDirectory(Name, NewFiles) then begin
          if not MetadataIsUnmodified(Name, OldFiles) then begin
            Result := 'Existing MyGPR version files were modified: ' + Name + '. Choose a new installation folder.';
            Exit;
          end;
        end;
      until not FindNext(Found);
    finally
      FindClose(Found);
    end;
  end;
end;

procedure RemoveObsoleteManagedFiles;
var
  OldFiles, NewFiles: TArrayOfString;
  I, J, Separator: Integer;
  Entry, RelativeName, OldHash, FileName: String;
  StillShipped: Boolean;
  NewNames: TStringList;
  Found: TFindRec;
begin
  if not LoadMigrationFiles(OldFiles, NewFiles) then Exit;
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
        if CompareText(GetSHA256OfFile(FileName), OldHash) = 0 then begin
          if DeleteFile(FileName) then
            RemoveEmptyManagedParents(FileName)
          else
            Log('Preserved obsolete file that could not be deleted: ' + RelativeName);
        end;
      except
        Log('Preserved obsolete file that could not be verified: ' + RelativeName);
      end;
    end;
  end;
  finally
    NewNames.Free;
  end;
  { Empty distribution directories also participate in Python version discovery. }
  if FindFirst(ExpandConstant('{app}\_internal\mygpr-*.dist-info'), Found) then begin
    try
      repeat
        RelativeName := '_internal\' + Found.Name;
        if ((Found.Attributes and $10) <> 0) and
           not RecordedDirectory(RelativeName, NewFiles) and SafeManagedPath(RelativeName) then
          RemoveDir(ExpandConstant('{app}\') + RelativeName);
      until not FindNext(Found);
    finally
      FindClose(Found);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then RemoveObsoleteManagedFiles;
end;

{ User projects, acquisition files and ~/MyGPR/config are outside
  the installer file table. Uninstall removes installed files, not user data. }
