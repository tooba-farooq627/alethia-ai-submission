# Full-run hardware and measured timings

Dataset revision: `84c74091e1c6ee7a5dffabfafb5c9033e4718883`

Started (UTC): 2026-11-20T05:48:56.648043+00:00
Completed (UTC): 2026-11-21T12:42:41.925582+00:00

Elapsed start-to-finish, including interruptions: 111225.278 seconds.

Recorded active session time, including downloads/checkpoint work: 64098.452 seconds.

Completed shards: 50. Videos: 1650. Output seconds: 461757. Video errors: 0.

Results SHA-256: `6f563e4a71b6855c4aacf1f9af1b390e9f8928c5568a974326fdf20c2b9870b2`

The first FP32 full run did not complete; no completed baseline full-run time is available. Subset projections are not full-run measurements.

The following inventories are the actual recorded full-run sessions. Disk inventory is reported as observed; cloud backing storage is not assumed.

## Recorded session 1

gpu:
```text
Tesla T4, 15360 MiB, 580.82.07
```

cpu:
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  2
On-line CPU(s) list:                     0,1
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz
CPU family:                              6
Model:                                   85
Thread(s) per core:                      2
Core(s) per socket:                      1
Socket(s):                               1
Stepping:                                3
BogoMIPS:                                4000.30
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               32 KiB (1 instance)
L1i cache:                               32 KiB (1 instance)
L2 cache:                                1 MiB (1 instance)
L3 cache:                                38.5 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0,1
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Vulnerable
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Vulnerable; SMT Host state unknown
Vulnerability Meltdown:                  Vulnerable
Vulnerability Mmio stale data:           Vulnerable
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Vulnerable
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Vulnerable: __user pointer sanitization and usercopy barriers only; no swapgs barriers
Vulnerability Spectre v2:                Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-eIBRS: Not affected; BHI: Vulnerable
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Vulnerable
Vulnerability Vmscape:                   Not affected
```

logical_cpu_count:
```text
2
```

ram:
```text
total        used        free      shared  buff/cache   available
Mem:     13605818368  1721516032   261722112     3133440 11959250944 11884302336
Swap:              0           0           0
```

disks:
```text
NAME  MODEL            ROTA  SIZE
loop0                     0  115G
sda   PersistentDisk      0  125G
sdb   PersistentDisk      0   46G
```

os:
```text
Linux-6.6.159+-x86_64-with-glibc2.39
```

python:
```text
3.11.14
```

## Recorded session 2

gpu:
```text
Tesla T4, 15360 MiB, 580.82.07
```

cpu:
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  2
On-line CPU(s) list:                     0,1
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz
CPU family:                              6
Model:                                   85
Thread(s) per core:                      2
Core(s) per socket:                      1
Socket(s):                               1
Stepping:                                3
BogoMIPS:                                4000.30
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               32 KiB (1 instance)
L1i cache:                               32 KiB (1 instance)
L2 cache:                                1 MiB (1 instance)
L3 cache:                                38.5 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0,1
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Vulnerable
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Vulnerable; SMT Host state unknown
Vulnerability Meltdown:                  Vulnerable
Vulnerability Mmio stale data:           Vulnerable
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Vulnerable
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Vulnerable: __user pointer sanitization and usercopy barriers only; no swapgs barriers
Vulnerability Spectre v2:                Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-eIBRS: Not affected; BHI: Vulnerable
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Vulnerable
Vulnerability Vmscape:                   Not affected
```

logical_cpu_count:
```text
2
```

ram:
```text
total        used        free      shared  buff/cache   available
Mem:     13605818368  1710116864  2783977472     3158016  9448353792 11895701504
Swap:              0           0           0
```

disks:
```text
NAME  MODEL            ROTA  SIZE
loop0                     0  115G
sda   PersistentDisk      0  125G
sdb   PersistentDisk      0   46G
```

os:
```text
Linux-6.6.159+-x86_64-with-glibc2.39
```

python:
```text
3.11.14
```

## Recorded session 3

gpu:
```text
Tesla T4, 15360 MiB, 580.82.07
```

cpu:
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  2
On-line CPU(s) list:                     0,1
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz
CPU family:                              6
Model:                                   85
Thread(s) per core:                      2
Core(s) per socket:                      1
Socket(s):                               1
Stepping:                                3
BogoMIPS:                                4000.30
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               32 KiB (1 instance)
L1i cache:                               32 KiB (1 instance)
L2 cache:                                1 MiB (1 instance)
L3 cache:                                38.5 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0,1
Vulnerability Gather data sampling:      Not affected
Vulnerability Indirect target selection: Vulnerable
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Vulnerable; SMT Host state unknown
Vulnerability Meltdown:                  Vulnerable
Vulnerability Mmio stale data:           Vulnerable
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Vulnerable
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Vulnerable
Vulnerability Spectre v1:                Vulnerable: __user pointer sanitization and usercopy barriers only; no swapgs barriers
Vulnerability Spectre v2:                Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-eIBRS: Not affected; BHI: Vulnerable
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Vulnerable
Vulnerability Vmscape:                   Not affected
```

logical_cpu_count:
```text
2
```

ram:
```text
total        used        free      shared  buff/cache   available
Mem:     13605818368  1709383680   260411392     3158016 11972653056 11896434688
Swap:              0           0           0
```

disks:
```text
NAME  MODEL            ROTA  SIZE
loop0                     0  115G
sda   PersistentDisk      0  125G
sdb   PersistentDisk      0   46G
```

os:
```text
Linux-6.6.159+-x86_64-with-glibc2.39
```

python:
```text
3.11.14
```

## Recorded session 4

gpu:
```text
Tesla T4, 15360 MiB, 580.178.04
Tesla T4, 15360 MiB, 580.178.04
```

cpu:
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz
CPU family:                              6
Model:                                   85
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                3
BogoMIPS:                                4000.38
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2 MiB (2 instances)
L3 cache:                                38.5 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Meltdown:                  Mitigation; PTI
Vulnerability Mmio stale data:           Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Mitigation; IBRS
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; IBRS; IBPB conditional; STIBP conditional; RSB filling; PBRSB-eIBRS Not affected; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Vmscape:                   Not affected
```

logical_cpu_count:
```text
4
```

ram:
```text
total        used        free      shared  buff/cache   available
Mem:     33658159104  1847738368 11524038656     1826816 20765696000 31810420736
Swap:              0           0           0
```

disks:
```text
NAME  MODEL            ROTA  SIZE
loop0                     0    1G
loop1                     0   96G
loop2                     0   20G
sda   PersistentDisk      0  128G
sdb   PersistentDisk      0    8T
```

os:
```text
Linux-6.18.55+-x86_64-with-glibc2.39
```

python:
```text
3.11.14
```

## Recorded session 5

gpu:
```text
Tesla T4, 15360 MiB, 580.178.04
Tesla T4, 15360 MiB, 580.178.04
```

cpu:
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz
CPU family:                              6
Model:                                   85
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                3
BogoMIPS:                                4000.40
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2 MiB (2 instances)
L3 cache:                                38.5 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Meltdown:                  Mitigation; PTI
Vulnerability Mmio stale data:           Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Mitigation; IBRS
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; IBRS; IBPB conditional; STIBP conditional; RSB filling; PBRSB-eIBRS Not affected; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Vmscape:                   Not affected
```

logical_cpu_count:
```text
4
```

ram:
```text
total        used        free      shared  buff/cache   available
Mem:     33658155008  1612161024 12251041792     1830912 20274262016 32045993984
Swap:              0           0           0
```

disks:
```text
NAME  MODEL            ROTA  SIZE
loop0                     0    1G
loop1                     0   96G
loop2                     0   20G
sda   PersistentDisk      0  128G
sdb   PersistentDisk      0    8T
```

os:
```text
Linux-6.18.55+-x86_64-with-glibc2.39
```

python:
```text
3.11.14
```

## Recorded session 6

gpu:
```text
Tesla T4, 15360 MiB, 580.178.04
Tesla T4, 15360 MiB, 580.178.04
```

cpu:
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz
CPU family:                              6
Model:                                   85
Thread(s) per core:                      2
Core(s) per socket:                      2
Socket(s):                               1
Stepping:                                3
BogoMIPS:                                4000.38
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               64 KiB (2 instances)
L1i cache:                               64 KiB (2 instances)
L2 cache:                                2 MiB (2 instances)
L3 cache:                                38.5 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Mitigation; Aligned branch/return thunks
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Mitigation; PTE Inversion
Vulnerability Mds:                       Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Meltdown:                  Mitigation; PTI
Vulnerability Mmio stale data:           Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Mitigation; IBRS
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; IBRS; IBPB conditional; STIBP conditional; RSB filling; PBRSB-eIBRS Not affected; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Vmscape:                   Not affected
```

logical_cpu_count:
```text
4
```

ram:
```text
total        used        free      shared  buff/cache   available
Mem:     33658155008  1626693632 12168568832     1839104 20342218752 32031461376
Swap:              0           0           0
```

disks:
```text
NAME  MODEL            ROTA  SIZE
loop0                     0    1G
loop1                     0   96G
loop2                     0   20G
sda   PersistentDisk      0  128G
sdb   PersistentDisk      0    8T
```

os:
```text
Linux-6.18.55+-x86_64-with-glibc2.39
```

python:
```text
3.11.14
```
