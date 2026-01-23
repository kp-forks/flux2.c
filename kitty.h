/*
 * kitty.h - Terminal graphics protocol support (Kitty and iTerm2)
 */

#ifndef KITTY_H
#define KITTY_H

#include "flux.h"

/* Terminal graphics protocol types */
typedef enum {
    TERM_PROTO_KITTY,   /* Kitty graphics protocol */
    TERM_PROTO_ITERM2   /* iTerm2 inline image protocol */
} term_graphics_proto;

/*
 * Display image in terminal using Kitty graphics protocol.
 * Returns 0 on success, -1 on error.
 */
int kitty_display_image(const flux_image *img);

/*
 * Display PNG file in terminal using Kitty graphics protocol.
 * Returns 0 on success, -1 on error.
 */
int kitty_display_png(const char *path);

/*
 * Display PNG file in terminal using iTerm2 inline image protocol.
 * Format: \033]1337;File=inline=1:<base64>\a
 * Returns 0 on success, -1 on error.
 */
int iterm2_display_png(const char *path);

/*
 * Display PNG file using the specified protocol.
 * Returns 0 on success, -1 on error.
 */
int terminal_display_png(const char *path, term_graphics_proto proto);

#endif /* KITTY_H */
