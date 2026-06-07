# fusekit

Apple Metal megakernel compiler for mlx-lm models.

## Status

Pre-alpha. CLI scaffold only.

## Commands

```bash
fusekit ls                  # list oMLX models with patch status
fusekit patch <model>       # compile megakernel + install mlx-lm hook
fusekit unpatch <model>     # remove compiled kernel + uninstall hook if last
fusekit status              # overall fusekit state
```

## Build

```bash
cargo build --release
./target/release/fusekit --help
```
