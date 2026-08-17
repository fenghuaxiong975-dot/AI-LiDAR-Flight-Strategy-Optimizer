# Model checkpoint

The desktop application expects the trained checkpoint at:

`models/latest_model-new.t7`

The supplied checkpoint is about 33 MB. It is included in the **full local archive** prepared from the original project, but excluded from the **browser-upload archive** because GitHub's normal browser file upload has a lower per-file limit.

For a public GitHub repository, either:

1. push the checkpoint with Git/Git LFS, or
2. attach it to a GitHub Release and ask users to download it into this directory.
