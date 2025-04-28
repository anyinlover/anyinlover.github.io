---
tags:
  - Linux
---

# Chapter15  Storage Media

* mount - Mount a file system
* umount - unmount a file system
* fsck - Check and repair a file system
* fdisk - Partition table manipulator
* mkfs - Create a file system
* fdformat - format a floppy disk
* dd - Write block oriented data directory to a device
* genisoimage (mkisofs) - Create an ISO 9660 image file
* wodim (cdrecord) - Write data to optical storage media
* md5sum - Calculate an MD5 checksum

## Mounting And Unmounting Storage Devices

A file named /etc/fstab lists the devices (hard disk partitions) that are to be mounted at boot time.

/etc/fstab Fields

| Field |     Contents     | Description                              |
| :---: | :--------------: | :--------------------------------------- |
|   1   |      Device      | A text label                             |
|   2   |   Mount Point    | The directory where the device is attached to the file system tree |
|   3   | File System Type | Linux allows many file system types to be mounted |
|   4   |     Options      | File systems can be mounted with various options |
|   5   |    Frequency     | A single number that specified if and when a file system is to be backed up with the dump command |
|   6   |      Order       | A single number that specifies in what order file systems should be checked with the fsck command |

### viewing A List Of Mounted File Systems

`mount`

unmount the disc

`sudo umount /dev/hdc`

mount the CD-ROM at the new mount point

`sudo mount -t iso9660 /dev/hdc /mnt/cdrom`

`-t` option is used to specify the file system type

### Determining Device names

Look at how the system names devices

`ls /dev`

Linux Storage Device Names

| Pattern  | Device                                   |
| :------: | :--------------------------------------- |
| /dev/fd* | Floppy disk drives                       |
| /dev/hd* | IDE (PATA) diska on older systems        |
| /dev/lp* | Printers                                 |
| /dev/sd* | SCSI disks                               |
| /dev/sr* | Optical drivers (CD/DVD readers and burners) |

Start a real-time view of the /var/log/syslog file

`sudo tail -f /var/log/syslog`

Show information about the file system

`df`

## Creating New File Systems

### Manipulating Partitions With fdisk

Must first unmount it

`sudo fdisk /dev/sdb`

### Creating A New File System With mkfs

~~~shell
sudo mkfs -t ext3 /dec/sdb1
sudo mkfs -t vfat /dev/sdb1
~~~

## Testing And Repairing File Systems

`sudo fsck /dev/sdb1`

## Formatting Floppy Disks

~~~shell
sudo fdformat /dev/fd0
sudo mkfs -t msdos /dev/fd0
~~~

## Moving Data Directly to/From Devices

~~~shell
dd if=input_file of=output_file [bs=block_size [count=blocks]]
dd if=/dev/sdb of=/dev/sdc
dd if=/dev/sdb of=flash_drive.img
~~~

## Creating CD-ROM Images

### Creating An Image Copy Of A CD-ROM

`dd if=/dev/cdrom of=ubuntu.iso`

### Creating An Image From A Collection Of Files

`genisoimage -o cd-rom.iso -R -J ~/cd-rom-files`

## Writing CD-ROM Images

### Mounting An ISO Image Directly

`mount -t iso9660 -o loop image.iso /mnt/iso_image`

### Blanking A Re-Writable CD-ROM

`wodim dev=/dev/cdrw blank=fast`

### Writing An Image

`wodim dev=/dev/cdrw image.iso`

## Extra Credit

`md5sum image.iso`
