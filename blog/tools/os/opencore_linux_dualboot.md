---
title: OpenCore Linux DualBoot Setting
date: 2025-02-27 14:43:00
tags:
  - OpenCore
  - Linux
  - BootLoader
---

As Apple and Homebrew ceased support for my 2015 MacBook Pro, I turned to OpenCore Patcher to ensure continued functionality. However, recently, many Python packages (pypi) no longer support Intel macOS versions, leading me to free approximately 400GB of disk space and instal Linux Mint alongside macOS.

Upon installation, the OpenCore Patcher boot menu unexpectedly disappeared after rebooting into Linux Mint. This issue arises because Linux Mint put it the first one in the boot order. Each operating system—whether it’s macOS or Linux—installs its own bootloader in the EFI partition. In my case, the EFI partition was mounted under `/boot/efi` when in Linux Mint.

To resolve this, I followed the easiest way in [OpenCore MultiBoot](https://dortania.github.io/OpenCore-Multiboot/oc/linux.html)

1. **Backup and Modify the OpenCore Config File:**
   - The original configuration file for OpenCore is located at `/boot/efi/EFI/OC/config.plist`.
   - Backup this file before making any changes.
   - Navigate to `Misc -> BlessOverride` within the config.plist editor.
   - Change the value from `\EFI\Microsoft\Boot\bootmgfw.efi` to `\EFI\ubuntu\grubx64.efi`.

2. **Restart and Access OpenCore Menu:**
   - After making these changes, restart your computer while holding down the Option key.
   - From the boot menu, select "OpenCore" to access the EFI menu.

3. **Set OpenCore as Default Bootloader:**
   - Once inside the OpenCore menu, it will automatically set as the first entry in the boot order.
   - This means you won’t need to press Option every time you reboot your system.

By following these steps, I have successfully restored the functionality of my dual-boot setup, ensuring a seamless transition between macOS and Linux Mint while retaining control over which operating system boots by default.
