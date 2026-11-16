use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};
use hound;
use std::f64::consts::PI;

const DISPLAY_POINTS: usize = 2000;
const PHASE_STEPS: usize = 36;
const DEFAULT_F0: f64 = 440.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BasisType {
    Sine,
    Triangle,
    Square,
    Sawtooth,
}

impl BasisType {
    const ALL: [BasisType; 4] = [
        BasisType::Sine,
        BasisType::Triangle,
        BasisType::Square,
        BasisType::Sawtooth,
    ];

    fn label(self) -> &'static str {
        match self {
            BasisType::Sine => "Sine (orthogonal)",
            BasisType::Triangle => "Triangle",
            BasisType::Square => "Square",
            BasisType::Sawtooth => "Sawtooth",
        }
    }

    fn eval(self, t: f64) -> f64 {
        match self {
            BasisType::Sine => (2.0 * PI * t).sin(),
            BasisType::Triangle => 2.0 / PI * (2.0 * PI * t).sin().asin(),
            BasisType::Square => (2.0 * PI * t).sin().signum(),
            BasisType::Sawtooth => 2.0 * (t - (t + 0.5).floor()),
        }
    }
}

struct Signal {
    samples: Vec<f32>,
    sample_rate: u32,
    path: String,
}

fn load_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };

    let mut mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|c| c[0])
            .collect()
    } else {
        samples
    };

    let max_amp = mono.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_amp > 1e-12 {
        for s in &mut mono {
            *s /= max_amp;
        }
    }

    Ok((mono, spec.sample_rate))
}

struct AnalysisResult {
    coefficients: Vec<f64>,
    original_windowed: Vec<f64>,
    reconstruction: Vec<f64>,
    energy_captured: f64,
}

fn analyze(
    samples: &[f32],
    sample_rate: u32,
    basis: BasisType,
    f0: f64,
    k_count: usize,
    window_secs: f64,
) -> AnalysisResult {
    let sr = sample_rate as f64;
    let window_samples = (window_secs * sr) as usize;

    let peak_idx = samples
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let start = peak_idx.saturating_sub(window_samples / 2);
    let end = (start + window_samples).min(samples.len());
    let n = end.saturating_sub(start);
    if n == 0 {
        return AnalysisResult {
            coefficients: Vec::new(),
            original_windowed: Vec::new(),
            reconstruction: Vec::new(),
            energy_captured: 0.0,
        };
    }

    let x: Vec<f64> = samples[start..end]
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let hann = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1).max(1) as f64).cos());
            s as f64 * hann
        })
        .collect();

    let phases: Vec<f64> = (0..n).map(|i| i as f64 / sr * f0).collect();

    let mut coefficients = Vec::with_capacity(k_count);
    let mut reconstruction = vec![0.0f64; n];
    let mut buffer = vec![0.0f64; n];

    for k in 1..=k_count {
        let kf = k as f64;
        let mut best_c = 0.0f64;
        let mut best_shift = 0.0f64;

        for step in 0..PHASE_STEPS {
            let shift = step as f64 / PHASE_STEPS as f64;
            for i in 0..n {
                buffer[i] = basis.eval((kf * phases[i] + shift).fract());
            }
            let num: f64 = x.iter().zip(&buffer).map(|(a, b)| a * b).sum();
            let den: f64 = buffer.iter().map(|b| b * b).sum();
            let c = if den > 1e-12 { num / den } else { 0.0 };
            if c.abs() > best_c.abs() {
                best_c = c;
                best_shift = shift;
            }
        }

        coefficients.push(best_c);

        for i in 0..n {
            reconstruction[i] += best_c * basis.eval((kf * phases[i] + best_shift).fract());
        }
    }

    let sig_energy: f64 = x.iter().map(|v| v * v).sum();
    let err_energy: f64 = x
        .iter()
        .zip(&reconstruction)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let energy_captured = if sig_energy > 1e-12 {
        1.0 - err_energy / sig_energy
    } else {
        0.0
    };

    AnalysisResult {
        coefficients,
        original_windowed: x,
        reconstruction,
        energy_captured,
    }
}

fn detect_fundamental(samples: &[f32], sample_rate: u32) -> f64 {
    let sr = sample_rate as f64;
    let window = samples.len().min(16384);
    if window < 1000 {
        return DEFAULT_F0;
    }

    let min_lag = (sr / 4000.0).max(1.0) as usize;
    let max_lag = ((sr / 50.0) as usize).min(window / 2);
    if max_lag <= min_lag + 2 {
        return DEFAULT_F0;
    }

    let mut corrs = vec![0.0f64; max_lag + 2];
    for lag in min_lag..max_lag {
        let mut corr = 0.0f64;
        let mut norm1 = 0.0f64;
        let mut norm2 = 0.0f64;
        for i in 0..window - lag {
            let a = samples[i] as f64;
            let b = samples[i + lag] as f64;
            corr += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }
        corrs[lag] = if norm1 > 1e-12 && norm2 > 1e-12 {
            corr / (norm1 * norm2).sqrt()
        } else {
            0.0
        };
    }

    let global_max = corrs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if global_max < 0.3 {
        return DEFAULT_F0;
    }

    let threshold = global_max * 0.9;
    for lag in (min_lag + 1)..max_lag {
        if corrs[lag] > threshold && corrs[lag] >= corrs[lag - 1] && corrs[lag] >= corrs[lag + 1] {
            return sr / lag as f64;
        }
    }
    sr / min_lag as f64
}

fn downsample(samples: &[f64], target: usize) -> Vec<f64> {
    if samples.len() <= target {
        return samples.to_vec();
    }
    let step = samples.len() as f64 / target as f64;
    (0..target)
        .map(|i| {
            let start = (i as f64 * step) as usize;
            let end = (((i + 1) as f64 * step) as usize).min(samples.len());
            if end <= start {
                return 0.0;
            }
            samples[start..end].iter().sum::<f64>() / (end - start) as f64
        })
        .collect()
}

fn to_pts(v: &[f64], spacing: f64) -> Vec<[f64; 2]> {
    v.iter()
        .enumerate()
        .map(|(i, &y)| [i as f64 * spacing, y])
        .collect()
}

struct App {
    signal: Option<Signal>,
    basis_type: BasisType,
    num_harmonics: usize,
    fundamental: f64,
    window_secs: f64,
    result: Option<AnalysisResult>,
    needs_analysis: bool,
    error: Option<String>,
    show_reconstruction: bool,
}

impl App {
    fn new() -> Self {
        Self {
            signal: None,
            basis_type: BasisType::Sine,
            num_harmonics: 64,
            fundamental: DEFAULT_F0,
            window_secs: 0.2,
            result: None,
            needs_analysis: false,
            error: None,
            show_reconstruction: true,
        }
    }

    fn load(&mut self, path: &str) {
        match load_wav(path) {
            Ok((samples, sample_rate)) => {
                self.signal = Some(Signal {
                    samples,
                    sample_rate,
                    path: path.to_string(),
                });
                self.needs_analysis = true;
                self.error = None;
            }
            Err(e) => self.error = Some(format!("Failed to load: {e}")),
        }
    }

    fn run_analysis(&mut self) {
        let Some(sig) = &self.signal else { return };
        self.result = Some(analyze(
            &sig.samples,
            sig.sample_rate,
            self.basis_type,
            self.fundamental,
            self.num_harmonics,
            self.window_secs,
        ));
        self.needs_analysis = false;
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Load WAV…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("WAV audio", &["wav"])
                        .pick_file()
                    {
                        if let Some(p) = path.to_str() {
                            self.load(p);
                        }
                    }
                }
                if let Some(sig) = &self.signal {
                    ui.label(format!(
                        "{}   |   {:.2} s  @  {} Hz",
                        sig.path,
                        sig.samples.len() as f64 / sig.sample_rate as f64,
                        sig.sample_rate
                    ));
                } else {
                    ui.weak("No file loaded");
                }
            });
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
            }
        });

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Analysis");
                ui.separator();

                egui::ComboBox::from_label("Basis")
                    .selected_text(self.basis_type.label())
                    .show_ui(ui, |ui| {
                        for bt in BasisType::ALL {
                            if ui
                                .selectable_value(&mut self.basis_type, bt, bt.label())
                                .clicked()
                            {
                                self.needs_analysis = true;
                            }
                        }
                    });

                if ui
                    .add(egui::Slider::new(&mut self.num_harmonics, 1..=256).text("Harmonics"))
                    .changed()
                {
                    self.needs_analysis = true;
                }

                if ui
                    .add(
                        egui::Slider::new(&mut self.window_secs, 0.01..=5.0)
                            .logarithmic(true)
                            .text("Window (s)"),
                    )
                    .changed()
                {
                    self.needs_analysis = true;
                }

                if ui
                    .add(
                        egui::Slider::new(&mut self.fundamental, 20.0..=4000.0)
                            .logarithmic(true)
                            .text("f₀ (Hz)"),
                    )
                    .changed()
                {
                    self.needs_analysis = true;
                }

                if ui.button("Auto-detect f₀").clicked() {
                    if let Some(sig) = &self.signal {
                        self.fundamental = detect_fundamental(&sig.samples, sig.sample_rate);
                        self.needs_analysis = true;
                    }
                }

                ui.separator();
                ui.checkbox(&mut self.show_reconstruction, "Show reconstruction");

                if self.needs_analysis && self.signal.is_some() {
                    self.run_analysis();
                }

                if let Some(res) = &self.result {
                    ui.separator();
                    ui.heading("Result");
                    let pct = (res.energy_captured * 100.0).clamp(-999.0, 100.0);
                    ui.label(format!("Energy captured: {:.1}%", pct));

                    if self.basis_type != BasisType::Sine {
                        ui.add_space(4.0);
                        ui.small(
                            egui::RichText::new(
                                "Non-orthogonal coefficients are similarity \
                                 scores, not energy decomposition",
                            )
                            .italics()
                            .color(egui::Color32::from_rgb(200, 170, 80)),
                        );
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("Coefficients").strong());
                    egui::ScrollArea::vertical()
                        .max_height(350.0)
                        .show(ui, |ui| {
                            for (k, &c) in res.coefficients.iter().enumerate() {
                                let f = self.fundamental * (k as f64 + 1.0);
                                ui.monospace(format!(
                                    "H{:>3}  {:>7.1} Hz   |c|={:.4}",
                                    k + 1,
                                    f,
                                    c.abs()
                                ));
                            }
                        });
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (Some(sig), Some(res)) = (&self.signal, &self.result) else {
                ui.centered_and_justified(|ui| ui.label("No wav file loaded"));
                return;
            };
            if res.original_windowed.is_empty() {
                return;
            }

            let actual_duration = res.original_windowed.len() as f64 / sig.sample_rate as f64;
            let display = downsample(&res.original_windowed, DISPLAY_POINTS);
            let display_recon = downsample(&res.reconstruction, DISPLAY_POINTS);
            let spacing = actual_duration / display.len() as f64;

            let avail = ui.available_height();
            let plot_height = ((avail - 56.0) / 2.0).max(120.0);

            ui.heading("waveform");
            Plot::new("waveform")
                .height(plot_height)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(to_pts(&display, spacing)))
                            .name("Original")
                            .width(1.0f32),
                    );
                    if self.show_reconstruction {
                        plot_ui.line(
                            Line::new(PlotPoints::from(to_pts(&display_recon, spacing)))
                                .name(format!("Reconstruction ({})", self.basis_type.label()))
                                .width(1.5f32),
                        );
                    }
                });

            ui.add_space(8.0);
            ui.heading(format!("coefficients - {} basis", self.basis_type.label()));

            let bars: Vec<Bar> = res
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, &c)| Bar::new((i + 1) as f64, c.abs()))
                .collect();

            Plot::new("coefficients")
                .height(plot_height)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new(bars).name("|c_k|"));
                });
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Custom Basis Fourier Analysis"),
        ..Default::default()
    };
    eframe::run_native(
        "Custom Basis Fourier Analysis",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
