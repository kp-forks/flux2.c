/*
 * kitty.c - Terminal graphics protocol support (Kitty and iTerm2)
 *
 * Kitty graphics protocol:
 *   Format: \033_G<control>;<base64-payload>\033\\
 *   For large images, data is sent in chunks with m=1 (more) or m=0 (last).
 *
 * iTerm2 inline image protocol:
 *   Format: \033]1337;File=inline=1:<base64-payload>\a
 *   See: https://iterm2.com/documentation-images.html
 */

#include "kitty.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Base64 encoding table */
static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* Base64 encode data, returns malloc'd string (caller must free) */
static char *base64_encode(const unsigned char *data, size_t len, size_t *out_len) {
    size_t encoded_len = 4 * ((len + 2) / 3);
    char *encoded = malloc(encoded_len + 1);
    if (!encoded) return NULL;

    size_t i, j;
    for (i = 0, j = 0; i < len; ) {
        uint32_t a = i < len ? data[i++] : 0;
        uint32_t b = i < len ? data[i++] : 0;
        uint32_t c = i < len ? data[i++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;

        encoded[j++] = b64_table[(triple >> 18) & 0x3F];
        encoded[j++] = b64_table[(triple >> 12) & 0x3F];
        encoded[j++] = b64_table[(triple >> 6) & 0x3F];
        encoded[j++] = b64_table[triple & 0x3F];
    }

    /* Add padding */
    int pad = len % 3;
    if (pad) {
        encoded[encoded_len - 1] = '=';
        if (pad == 1) encoded[encoded_len - 2] = '=';
    }

    encoded[encoded_len] = '\0';
    if (out_len) *out_len = encoded_len;
    return encoded;
}

/*
 * Send PNG data using Kitty graphics protocol.
 * Data is sent in chunks to avoid terminal buffer issues.
 */
static int kitty_send_png(const unsigned char *png_data, size_t png_size) {
    /* Base64 encode the PNG data */
    size_t b64_len;
    char *b64_data = base64_encode(png_data, png_size, &b64_len);
    if (!b64_data) return -1;

    /* Send in chunks (4096 bytes of base64 per chunk is safe) */
    const size_t chunk_size = 4096;
    size_t offset = 0;
    int first = 1;

    while (offset < b64_len) {
        size_t remaining = b64_len - offset;
        size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
        int more = (offset + this_chunk) < b64_len;

        if (first) {
            /* First chunk: a=T (transmit+display), f=100 (PNG), t=d (direct) */
            printf("\033_Ga=T,f=100,t=d,m=%d;", more ? 1 : 0);
            first = 0;
        } else {
            /* Continuation chunk */
            printf("\033_Gm=%d;", more ? 1 : 0);
        }

        /* Write the base64 chunk */
        fwrite(b64_data + offset, 1, this_chunk, stdout);
        printf("\033\\");

        offset += this_chunk;
    }

    fflush(stdout);
    free(b64_data);
    return 0;
}

int kitty_display_png(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "kitty: cannot open %s\n", path);
        return -1;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return -1;
    }

    /* Read PNG data */
    unsigned char *png_data = malloc(size);
    if (!png_data) {
        fclose(f);
        return -1;
    }

    if (fread(png_data, 1, size, f) != (size_t)size) {
        free(png_data);
        fclose(f);
        return -1;
    }
    fclose(f);

    int result = kitty_send_png(png_data, size);
    free(png_data);

    /* Print newline after image */
    printf("\n");

    return result;
}

int kitty_display_image(const flux_image *img) {
    if (!img || !img->data) return -1;

    /* Raw pixel data size */
    size_t data_size = (size_t)img->width * img->height * img->channels;

    /* Base64 encode the raw pixel data */
    size_t b64_len;
    char *b64_data = base64_encode(img->data, data_size, &b64_len);
    if (!b64_data) return -1;

    /* f=24 for RGB (3 channels), f=32 for RGBA (4 channels) */
    int fmt = (img->channels == 4) ? 32 : 24;

    /* Send in chunks */
    const size_t chunk_size = 4096;
    size_t offset = 0;
    int first = 1;

    while (offset < b64_len) {
        size_t remaining = b64_len - offset;
        size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;
        int more = (offset + this_chunk) < b64_len;

        if (first) {
            /* First chunk: a=T (transmit+display), f=24/32, s=width, v=height */
            printf("\033_Ga=T,f=%d,s=%d,v=%d,m=%d;",
                   fmt, img->width, img->height, more ? 1 : 0);
            first = 0;
        } else {
            printf("\033_Gm=%d;", more ? 1 : 0);
        }

        fwrite(b64_data + offset, 1, this_chunk, stdout);
        printf("\033\\");
        offset += this_chunk;
    }

    fflush(stdout);
    free(b64_data);
    printf("\n");
    return 0;
}

/*
 * Send PNG data using iTerm2 inline image protocol.
 * Format: \033]1337;File=inline=1:<base64>\a
 */
static int iterm2_send_png(const unsigned char *png_data, size_t png_size) {
    size_t b64_len;
    char *b64_data = base64_encode(png_data, png_size, &b64_len);
    if (!b64_data) return -1;

    /* iTerm2 protocol: OSC 1337 ; File=inline=1 : <base64> BEL */
    printf("\033]1337;File=inline=1:");
    fwrite(b64_data, 1, b64_len, stdout);
    printf("\a\n");
    fflush(stdout);

    free(b64_data);
    return 0;
}

int iterm2_display_png(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "iterm2: cannot open %s\n", path);
        return -1;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return -1;
    }

    /* Read PNG data */
    unsigned char *png_data = malloc(size);
    if (!png_data) {
        fclose(f);
        return -1;
    }

    if (fread(png_data, 1, size, f) != (size_t)size) {
        free(png_data);
        fclose(f);
        return -1;
    }
    fclose(f);

    int result = iterm2_send_png(png_data, size);
    free(png_data);
    return result;
}

int terminal_display_png(const char *path, term_graphics_proto proto) {
    switch (proto) {
        case TERM_PROTO_ITERM2:
            return iterm2_display_png(path);
        case TERM_PROTO_KITTY:
        default:
            return kitty_display_png(path);
    }
}
