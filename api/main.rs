use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use std::path::Path;
use serde_json::json;
use vercel_runtime::{run, Body, Error, Request, Response, StatusCode};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(handler).await
}

// Convert an image to grayscale
fn to_grayscale(img: &DynamicImage) -> GrayImage {
    img.to_luma8()
}

// Save image
fn save_image<P: AsRef<Path>>(img: &GrayImage, path: P) {
    img.save(path).expect("Failed to save image");
}

// Floyd-Steinberg Dithering
fn floyd_steinberg_dither(img: &GrayImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut img_buf = img.clone();

    for y in 0..height {
        for x in 0..width {
            let old_pixel = img_buf.get_pixel(x, y)[0] as i16;
            let new_pixel = if old_pixel < 128 { 0 } else { 255 };
            let error = old_pixel - new_pixel;
            img_buf.put_pixel(x, y, Luma([new_pixel as u8]));

            for (dx, dy, factor) in [
                (1, 0, 7.0 / 16.0),
                (-1, 1, 3.0 / 16.0),
                (0, 1, 5.0 / 16.0),
                (1, 1, 1.0 / 16.0),
            ] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let pos = (nx as u32, ny as u32);
                    let neighbor_val = img_buf.get_pixel(pos.0, pos.1)[0] as f32;
                    let new_val = (neighbor_val + (error as f32 * factor)).clamp(0.0, 255.0);
                    img_buf.put_pixel(pos.0, pos.1, Luma([new_val as u8]));
                }
            }
        }
    }

    img_buf
}

// Bayer Ordered Dithering (4x4 Matrix)
const BAYER4: [[u8; 4]; 4] = [
    [15, 135, 45, 165],
    [195, 75, 225, 105],
    [60, 180, 30, 150],
    [240, 120, 210, 90],
];

fn ordered_dither(img: &GrayImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut dithered = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let threshold = BAYER4[(y % 4) as usize][(x % 4) as usize];
            let pixel = img.get_pixel(x, y)[0];
            let new_val = if pixel > threshold { 255 } else { 0 };
            dithered.put_pixel(x, y, Luma([new_val]));
        }
    }

    dithered
}

// Atkinson Dithering
fn atkinson_dither(img: &GrayImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut img_buf = img.clone();

    for y in 0..height {
        for x in 0..width {
            let old_pixel = img_buf.get_pixel(x, y)[0] as i16;
            let new_pixel = if old_pixel < 128 { 0 } else { 255 };
            let error = (old_pixel - new_pixel) / 8;
            img_buf.put_pixel(x, y, Luma([new_pixel as u8]));

            for (dx, dy) in [
                (1, 0), (2, 0),
                (-1, 1), (0, 1), (1, 1),
                (0, 2),
            ] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let pos = (nx as u32, ny as u32);
                    let neighbor_val = img_buf.get_pixel(pos.0, pos.1)[0] as i16;
                    let new_val = (neighbor_val + error).clamp(0, 255);
                    img_buf.put_pixel(pos.0, pos.1, Luma([new_val as u8]));
                }
            }
        }
    }

    img_buf
}

// Basic Threshold
fn threshold_dither(img: &GrayImage, threshold: u8) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut dithered = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y)[0];
            let new_val = if pixel > threshold { 255 } else { 0 };
            dithered.put_pixel(x, y, Luma([new_val]));
        }
    }

    dithered
}

fn select_algorithm(alg_type: &str, img: ImageBuffer<Luma<u8>, Vec<u8>>) -> GrayImage {
    match alg_type {
        "floyd-steinberg" => {
            floyd_steinberg_dither(&img)
        },
        "ordered" => {
            ordered_dither(&img)
        },
        "atkinson" => {
            atkinson_dither(&img)
        },
        _ => {
            threshold_dither(&img, 128)
        }
    }
}

/* 
fn main() {
    let input_path: &str = "input/iStock-884221008.jpg";
    let base_img: DynamicImage = image::open(input_path).expect("Failed to load image");
    let gray: ImageBuffer<Luma<u8>, Vec<u8>> = to_grayscale(&base_img);

    let  alg_type = "OK";

    select_algorithm(alg_type,gray);

    /* 
    save_image(&floyd_steinberg_dither(&gray), "output/floydsteinberg.png");
    save_image(&ordered_dither(&gray), "output/ordered.png");
    save_image(&atkinson_dither(&gray), "output/atkinson.png");
    save_image(&threshold_dither(&gray, 128), "output/threshold.png");
    */
}
*/

