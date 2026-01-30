# Goth Demo Examples - CSV and SVG Output Formats

This document specifies how Goth demo examples should output results in CSV and SVG formats for visualization and analysis.

## Part 1: CSV Output Formats

### 1.1 General CSV Structure

All CSV outputs should follow these standards:
- UTF-8 encoding
- Standard comma delimiter
- Headers in first row
- Quoted fields containing special characters
- ISO 8601 timestamps where applicable
- Consistent precision (e.g., 6 decimal places for floats)

### 1.2 CSV Format by Example Type

#### 1.2.1 Iterative Methods (Newton-Raphson)
```
iteration,x_current,x_previous,f_x,df_x,error,converged
0,2.0,NaN,7.0,12.0,NaN,false
1,1.41667,2.0,0.00694,4.83333,0.58333,false
2,1.41421,1.41667,0.0000061,4.828,0.00246,true
```

**Fields:**
- `iteration`: Iteration counter (int)
- `x_current`: Current estimate (float)
- `x_previous`: Previous estimate (float)
- `f_x`: Function value at current x
- `df_x`: Derivative value at current x
- `error`: Absolute error |x_current - x_previous|
- `converged`: Boolean convergence flag

#### 1.2.2 ODE Integration (Runge-Kutta)
```
time,x,y,vx,vy,ke,pe,total_energy,step_size
0.0,1.0,0.0,0.0,1.0,0.5,0.5,1.0,0.01
0.01,1.00995,0.00995,0.0995,0.99975,0.499875,0.500125,1.0,0.01
0.02,1.01980,0.01980,0.198,0.99901,0.499505,0.500495,1.0,0.01
```

**Fields:**
- `time`: Current time value
- `x`, `y`: Position coordinates
- `vx`, `vy`: Velocity components
- `ke`: Kinetic energy
- `pe`: Potential energy
- `total_energy`: Total energy (for conservation checking)
- `step_size`: Adaptive step size used

#### 1.2.3 PDE Solutions (Laplace/Poisson 2D Grid)
```
x,y,phi,source,iteration
0.0,0.0,1.0,0.0,500
0.1,0.0,0.95432,0.0,500
0.2,0.0,0.87654,0.0,500
0.0,0.1,0.96789,0.1,500
```

**Fields:**
- `x`, `y`: Grid coordinates
- `phi`: Solution value (potential, temperature, etc.)
- `source`: Source term value (for Poisson)
- `iteration`: Iteration count at which output occurred

#### 1.2.4 Monte Carlo Simulation (Pi Estimation)
```
sample_count,pi_estimate,error,std_dev,ci_lower,ci_upper
1000,3.144,0.00242,0.0512,3.044,3.244
10000,3.14159,0.0000107,0.0161,3.109,3.173
100000,3.141592,0.0000011,0.00512,3.130,3.153
1000000,3.1415926,0.00000011,0.00162,3.138,3.144
```

**Fields:**
- `sample_count`: Number of samples
- `pi_estimate`: Current estimate of pi
- `error`: Error from true value (3.14159265...)
- `std_dev`: Standard deviation
- `ci_lower`, `ci_upper`: 95% confidence interval bounds

#### 1.2.5 Linear System Solution (Gaussian Elimination)
```
iteration,residual_norm,solution_norm,condition_number
0,15.234,NaN,NaN
1,2.1456,3.4521,12.45
2,0.00234,3.4589,12.45
3,0.00001,3.4590,12.45
```

**Fields:**
- `iteration`: Iteration number
- `residual_norm`: ||Ax - b|| (measure of solution quality)
- `solution_norm`: ||x||
- `condition_number`: Condition number of matrix

#### 1.2.6 Wave Equation (1D or 2D)
```
time,x,u,u_t,u_x,u_xx
0.0,0.0,1.0,0.0,0.0,0.0
0.0,0.1,0.99499,0.0,−0.05,−0.1
0.01,0.0,1.0,0.0,0.0,0.0
0.01,0.1,0.98994,−0.05,−0.05,−0.1
```

**Fields:**
- `time`: Current time
- `x`: Position (or x,y for 2D)
- `u`: Solution value
- `u_t`: Temporal derivative
- `u_x`: Spatial derivative (1st)
- `u_xx`: Spatial derivative (2nd)

#### 1.2.7 Eigenvalue Computation
```
iteration,lambda_1,lambda_2,lambda_3,residual_1,residual_2,residual_3,converged
0,1.5,NaN,NaN,2.34,NaN,NaN,false
1,4.875,NaN,NaN,0.234,NaN,NaN,false
2,5.0,NaN,NaN,0.00123,NaN,NaN,true
```

**Fields:**
- `iteration`: Iteration number
- `lambda_i`: i-th eigenvalue estimate
- `residual_i`: Residual for i-th eigenvalue
- `converged`: Overall convergence flag

#### 1.2.8 Optimization (Gradient Descent)
```
iteration,x1,x2,f_value,gradient_norm,step_size,line_search_evals
0,0.0,0.0,4.0,4.472,NaN,NaN
1,−0.1,−0.1,3.96,4.412,0.1,3
2,−0.198,−0.198,3.84,4.234,0.1,2
3,−0.291,−0.291,3.65,3.98,0.1,2
```

**Fields:**
- `iteration`: Iteration number
- `x1`, `x2`, ...: Decision variables
- `f_value`: Objective function value
- `gradient_norm`: ||∇f||
- `step_size`: Step size used
- `line_search_evals`: Number of function evaluations in line search

#### 1.2.9 Interpolation/Fitting Results
```
x_data,y_data,y_fit,residual,local_error
0.0,0.0,0.001,−0.001,0.0001
0.1,0.0995,0.0998,−0.0003,0.00001
0.2,0.1987,0.1985,0.0002,0.00002
```

**Fields:**
- `x_data`: Original x values
- `y_data`: Original y values
- `y_fit`: Fitted/interpolated values
- `residual`: Difference (y_data - y_fit)
- `local_error`: Local truncation error estimate

### 1.3 CSV Export Function Template (Pseudo-Goth)

```goth
fn export_csv_header(fields: Array[String]) -> String {
  join(fields, ",")
}

fn export_csv_row(values: Array[Float | Int | Bool | String]) -> String {
  mapped = map(values, fn(v: Float | Int | Bool | String) -> String {
    match v {
      String(s) => quote_if_needed(s)
      Float(f) => format_float(f, 6)
      Int(i) => to_string(i)
      Bool(b) => b ? "true" : "false"
    }
  })
  join(mapped, ",")
}

fn write_csv_file(filename: String, headers: Array[String], 
                  rows: Array[Array[Float | Int | Bool | String]]) -> IO[Unit] {
  handle = open_file(filename, "w")
  write_line(handle, export_csv_header(headers))
  foreach(rows, fn(row) {
    write_line(handle, export_csv_row(row))
  })
  close_file(handle)
}
```

---

## Part 2: SVG Output Formats

### 2.1 General SVG Structure

All SVG outputs should:
- Use standard SVG 1.1 specification
- Include viewBox for proper scaling
- Include descriptive titles and comments
- Use consistent styling
- Include legends where applicable
- Provide 600x400 minimum resolution (scalable via viewBox)
- Use readable fonts (Arial, sans-serif)

### 2.2 SVG Output Formats by Example Type

#### 2.2.1 Convergence Plot (Iterative Methods)

**File: `newton_raphson_convergence.svg`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <title>Newton-Raphson Convergence</title>
  <defs>
    <style>
      .grid-line { stroke: #e0e0e0; stroke-width: 1; }
      .axis { stroke: black; stroke-width: 2; }
      .axis-label { font: 14px Arial; text-anchor: middle; }
      .plot-line { stroke: #2196F3; stroke-width: 2; fill: none; }
      .error-line { stroke: #FF6B6B; stroke-width: 2; fill: none; }
      .legend { font: 12px Arial; }
      .grid-text { font: 12px Arial; text-anchor: end; }
    </style>
  </defs>

  <!-- Title -->
  <text x="400" y="30" style="font-size: 20px; font-weight: bold; text-anchor: center;">
    Newton-Raphson: Square Root Convergence
  </text>

  <!-- Grid and axes -->
  <g id="plot-area">
    <polyline class="grid-line" points="50,50 50,500 750,500"/>
    
    <!-- X-axis -->
    <line x1="50" y1="500" x2="750" y2="500" class="axis"/>
    <text x="400" y="540" class="axis-label">Iteration</text>
    
    <!-- Y-axis -->
    <line x1="50" y1="50" x2="50" y2="500" class="axis"/>
    <text x="20" y="275" class="axis-label" transform="rotate(-90 20 275)">Error</text>
    
    <!-- Grid lines and labels -->
    <line x1="50" y1="100" x2="750" y2="100" class="grid-line"/>
    <text x="45" y="105" class="grid-text">10^-2</text>
    
    <line x1="50" y1="150" x2="750" y2="150" class="grid-line"/>
    <text x="45" y="155" class="grid-text">10^-4</text>
    
    <line x1="50" y1="200" x2="750" y2="200" class="grid-line"/>
    <text x="45" y="205" class="grid-text">10^-6</text>
    
    <!-- Plot data points and lines -->
    <polyline class="error-line" points="70,450 170,250 270,100 370,50"/>
    
    <!-- Data point markers -->
    <circle cx="70" cy="450" r="4" fill="#FF6B6B"/>
    <circle cx="170" cy="250" r="4" fill="#FF6B6B"/>
    <circle cx="270" cy="100" r="4" fill="#FF6B6B"/>
    <circle cx="370" cy="50" r="4" fill="#FF6B6B"/>
  </g>

  <!-- Legend -->
  <g id="legend" transform="translate(500, 70)">
    <rect x="0" y="0" width="220" height="80" fill="white" stroke="black" stroke-width="1"/>
    <line x1="10" y1="20" x2="40" y2="20" class="error-line"/>
    <text x="50" y="25" class="legend">Error |x_k - x*|</text>
    
    <text x="10" y="45" class="legend">Converged in 3 iterations</text>
    <text x="10" y="65" class="legend">Final error: 1.23e-7</text>
  </g>

  <!-- Metadata -->
  <text x="10" y="580" style="font-size: 10px; fill: #666;">
    Generated by Goth numerical solver | Method: Newton-Raphson | Target: f(x)=x²-2
  </text>
</svg>
```

#### 2.2.2 Phase Space Plot (ODE Solutions)

**File: `trajectory_phase_space.svg`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">
  <title>Phase Space Trajectory</title>
  <defs>
    <style>
      .trajectory { stroke: #2196F3; stroke-width: 2; fill: none; }
      .energy-contour { stroke: #CCC; stroke-width: 1; fill: none; stroke-dasharray: 2,2; }
      .axis { stroke: black; stroke-width: 2; }
      .initial-point { fill: #4CAF50; r: 6; }
      .final-point { fill: #FF6B6B; r: 6; }
      .axis-label { font: 14px Arial; text-anchor: middle; }
    </style>
  </defs>

  <!-- Title -->
  <text x="400" y="30" style="font-size: 20px; font-weight: bold; text-anchor: center;">
    Phase Space Trajectory: Pendulum Motion
  </text>

  <!-- Energy contours (isoenergy lines) -->
  <circle cx="400" cy="400" r="100" class="energy-contour"/>
  <circle cx="400" cy="400" r="150" class="energy-contour"/>
  <circle cx="400" cy="400" r="200" class="energy-contour"/>

  <!-- Axes -->
  <line x1="100" y1="400" x2="700" y2="400" class="axis"/>
  <line x1="400" y1="100" x2="400" y2="700" class="axis"/>

  <!-- Axis labels -->
  <text x="720" y="410" class="axis-label">Position (x)</text>
  <text x="390" y="90" class="axis-label">Velocity (v)</text>

  <!-- Trajectory path -->
  <polyline class="trajectory" points="
    400,250
    450,280
    480,330
    490,380
    480,420
    450,450
    400,470
    350,450
    320,420
    310,380
    320,330
    350,280
    400,250
  "/>

  <!-- Initial and final points -->
  <circle cx="400" cy="250" class="initial-point"/>
  <text x="420" y="240" style="font-size: 12px;">Start</text>

  <circle cx="400" cy="250" class="final-point" r="3" fill="none" stroke="#FF6B6B" stroke-width="2"/>
  <text x="420" y="270" style="font-size: 12px;">End (10s)</text>

  <!-- Legend -->
  <g transform="translate(50, 50)">
    <rect x="0" y="0" width="200" height="100" fill="white" stroke="black" stroke-width="1"/>
    <polyline class="energy-contour" points="10,20 40,20"/>
    <text x="50" y="25" style="font-size: 12px;">Energy contours</text>
    <polyline class="trajectory" points="10,50 40,50"/>
    <text x="50" y="55" style="font-size: 12px;">System trajectory</text>
    <circle cx="25" cy="75" class="initial-point" r="4"/>
    <text x="50" y="80" style="font-size: 12px;">Initial condition</text>
  </g>

  <text x="10" y="750" style="font-size: 10px; fill: #666;">
    Simulation time: 10s | Energy (E) = 1.0 J
  </text>
</svg>
```

#### 2.2.3 Heat Map (2D PDE Solution)

**File: `laplace_solution_heatmap.svg`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 1000 600" xmlns="http://www.w3.org/2000/svg">
  <title>2D Laplace Equation Solution</title>
  <defs>
    <linearGradient id="colorGrad" x1="0%" y1="0%" x2="100%">
      <stop offset="0%" style="stop-color:#0000FF;stop-opacity:1"/>
      <stop offset="25%" style="stop-color:#00FFFF;stop-opacity:1"/>
      <stop offset="50%" style="stop-color:#00FF00;stop-opacity:1"/>
      <stop offset="75%" style="stop-color:#FFFF00;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#FF0000;stop-opacity:1"/>
    </linearGradient>
    <style>
      .cell { stroke: none; }
      .border { stroke: black; stroke-width: 2; fill: none; }
      .colorbar-label { font: 12px Arial; }
      .title { font: 18px Arial; font-weight: bold; }
    </style>
  </defs>

  <!-- Title -->
  <text x="400" y="30" class="title" text-anchor="middle">
    2D Laplace Equation: Temperature Distribution
  </text>

  <!-- Grid of colored cells (simplified representation) -->
  <g id="solution-grid">
    <!-- Row 1: cooler (blue) -->
    <rect x="50" y="60" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.3" class="cell"/>
    <rect x="80" y="60" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.35" class="cell"/>
    <rect x="110" y="60" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.4" class="cell"/>
    <!-- ... more cells ... -->
    
    <!-- Row middle: transition -->
    <rect x="50" y="150" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.5" class="cell"/>
    <rect x="80" y="150" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.55" class="cell"/>
    <rect x="110" y="150" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.6" class="cell"/>
    
    <!-- Row 3: warmer (red) -->
    <rect x="50" y="240" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.8" class="cell"/>
    <rect x="80" y="240" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.85" class="cell"/>
    <rect x="110" y="240" width="30" height="30" fill="url(#colorGrad)" fill-opacity="0.9" class="cell"/>
  </g>

  <!-- Domain border -->
  <rect x="50" y="60" width="300" height="210" class="border"/>

  <!-- Axis labels -->
  <text x="200" y="310" text-anchor="middle" style="font: 12px Arial;">x</text>
  <text x="20" y="165" text-anchor="middle" style="font: 12px Arial;">y</text>

  <!-- Colorbar -->
  <g id="colorbar" transform="translate(450, 60)">
    <defs>
      <linearGradient id="cbGrad" x1="0%" y1="100%" x2="0%" y2="0%">
        <stop offset="0%" style="stop-color:#0000FF;stop-opacity:1"/>
        <stop offset="25%" style="stop-color:#00FFFF;stop-opacity:1"/>
        <stop offset="50%" style="stop-color:#00FF00;stop-opacity:1"/>
        <stop offset="75%" style="stop-color:#FFFF00;stop-opacity:1"/>
        <stop offset="100%" style="stop-color:#FF0000;stop-opacity:1"/>
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="30" height="210" fill="url(#cbGrad)"/>
    <rect x="0" y="0" width="30" height="210" fill="none" stroke="black" stroke-width="1"/>
    
    <!-- Colorbar labels -->
    <text x="40" y="10" class="colorbar-label">100 (hot)</text>
    <text x="40" y="110" class="colorbar-label">50 (med)</text>
    <text x="40" y="215" class="colorbar-label">0 (cool)</text>
  </g>

  <!-- Statistics box -->
  <g transform="translate(450, 320)">
    <rect x="0" y="0" width="280" height="120" fill="white" stroke="black" stroke-width="1"/>
    <text x="10" y="25" style="font: 12px Arial; font-weight: bold;">Solution Statistics</text>
    <text x="10" y="50" style="font: 11px Arial;">Max temperature: 100.00 K</text>
    <text x="10" y="70" style="font: 11px Arial;">Min temperature: 0.00 K</text>
    <text x="10" y="90" style="font: 11px Arial;">Iterations: 500 | Residual: 1.23e-6</text>
    <text x="10" y="110" style="font: 11px Arial;">Grid resolution: 64x64 cells</text>
  </g>

  <text x="10" y="580" style="font-size: 10px; fill: #666;">
    Domain: [0,1]×[0,1] | Dirichlet BC: T=100 at x=1, T=0 elsewhere
  </text>
</svg>
```

#### 2.2.4 Multi-Series Line Plot (Multiple Solutions)

**File: `multi_comparison.svg`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 900 600" xmlns="http://www.w3.org/2000/svg">
  <title>Method Comparison Plot</title>
  <defs>
    <style>
      .series-1 { stroke: #2196F3; stroke-width: 2.5; fill: none; }
      .series-2 { stroke: #FF9800; stroke-width: 2.5; fill: none; stroke-dasharray: 5,5; }
      .series-3 { stroke: #4CAF50; stroke-width: 2.5; fill: none; stroke-dasharray: 10,3; }
      .reference { stroke: #333; stroke-width: 1; fill: none; stroke-dasharray: 2,2; }
      .axis { stroke: black; stroke-width: 2; }
      .grid { stroke: #e0e0e0; stroke-width: 1; }
      .legend { font: 12px Arial; }
      .axis-label { font: 14px Arial; }
    </style>
  </defs>

  <text x="450" y="30" style="font-size: 20px; font-weight: bold; text-anchor: center;">
    Method Comparison: RK4 vs Analytical Solution
  </text>

  <!-- Axes and grid -->
  <g id="plot">
    <line x1="80" y1="50" x2="80" y2="480" class="axis"/>
    <line x1="80" y1="480" x2="850" y2="480" class="axis"/>
    
    <!-- Horizontal grid lines -->
    <line x1="80" y1="150" x2="850" y2="150" class="grid"/>
    <line x1="80" y1="250" x2="850" y2="250" class="grid"/>
    <line x1="80" y1="350" x2="850" y2="350" class="grid"/>
    
    <!-- Vertical grid lines -->
    <line x1="200" y1="50" x2="200" y2="480" class="grid"/>
    <line x1="400" y1="50" x2="400" y2="480" class="grid"/>
    <line x1="600" y1="50" x2="600" y2="480" class="grid"/>
    <line x1="800" y1="50" x2="800" y2="480" class="grid"/>
    
    <!-- Series 1: RK4 solution -->
    <polyline class="series-1" points="
      100,420 120,400 140,380 160,350 180,320 200,280
      220,240 240,200 260,160 280,130 300,110 320,100
      340,100 360,110 380,130 400,160 420,200 440,240
      460,280 480,320 500,350 520,380 540,400 560,420
    "/>
    
    <!-- Series 2: Alternative method -->
    <polyline class="series-2" points="
      100,422 120,402 140,382 160,352 180,322 200,282
      220,242 240,202 260,162 280,132 300,110 320,102
      340,102 360,112 380,132 400,162 420,202 440,242
      460,282 480,322 500,352 520,382 540,402 560,422
    "/>
    
    <!-- Reference/Analytical solution -->
    <polyline class="reference" points="
      100,420 120,400 140,380 160,350 180,320 200,280
      220,240 240,200 260,160 280,130 300,110 320,100
      340,100 360,110 380,130 400,160 420,200 440,240
      460,280 480,320 500,350 520,380 540,400 560,420
    "/>
  </g>

  <!-- Axis labels -->
  <text x="450" y="520" text-anchor="middle" class="axis-label">Time (s)</text>
  <text x="20" y="265" text-anchor="middle" class="axis-label" transform="rotate(-90 20 265)">Displacement (m)</text>

  <!-- Y-axis tick labels -->
  <text x="70" y="485" text-anchor="end" style="font: 11px Arial;">0</text>
  <text x="70" y="355" text-anchor="end" style="font: 11px Arial;">0.5</text>
  <text x="70" y="225" text-anchor="end" style="font: 11px Arial;">1.0</text>
  <text x="70" y="95" text-anchor="end" style="font: 11px Arial;">1.5</text>

  <!-- Legend -->
  <g transform="translate(620, 100)">
    <rect x="0" y="0" width="200" height="120" fill="white" stroke="black" stroke-width="1"/>
    
    <polyline class="series-1" points="10,20 40,20"/>
    <text x="50" y="25" class="legend">RK4 (dt=0.01s)</text>
    
    <polyline class="series-2" points="10,50 40,50"/>
    <text x="50" y="55" class="legend">Euler (dt=0.01s)</text>
    
    <polyline class="reference" points="10,80 40,80"/>
    <text x="50" y="85" class="legend">Analytical</text>
    
    <text x="10" y="110" style="font: 10px Arial; font-weight: bold;">Max error: 1.2e-4</text>
  </g>
</svg>
```

#### 2.2.5 Convergence Rate Plot (Log Scale)

**File: `convergence_log_scale.svg`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <title>Convergence Rate (Log Scale)</title>
  <defs>
    <style>
      .theory { stroke: #999; stroke-width: 2; fill: none; stroke-dasharray: 5,5; }
      .method-1 { stroke: #2196F3; stroke-width: 2.5; fill: none; }
      .method-2 { stroke: #FF6B6B; stroke-width: 2.5; fill: none; }
      .marker { fill: #2196F3; r: 3; }
      .axis { stroke: black; stroke-width: 2; }
      .grid { stroke: #ccc; stroke-width: 1; }
    </style>
  </defs>

  <text x="400" y="30" style="font-size: 20px; font-weight: bold; text-anchor: center;">
    Convergence Rates (Log-Log Plot)
  </text>

  <!-- Log grid -->
  <g id="grid">
    <!-- Vertical lines (decade markers) -->
    <line x1="100" y1="50" x2="100" y2="480" class="grid"/>
    <line x1="200" y1="50" x2="200" y2="480" class="grid"/>
    <line x1="300" y1="50" x2="300" y2="480" class="grid"/>
    <line x1="400" y1="50" x2="400" y2="480" class="grid"/>
    <line x1="500" y1="50" x2="500" y2="480" class="grid"/>
    <line x1="600" y1="50" x2="600" y2="480" class="grid"/>
    <line x1="700" y1="50" x2="700" y2="480" class="grid"/>
    
    <!-- Horizontal lines (decade markers) -->
    <line x1="50" y1="100" x2="750" y2="100" class="grid"/>
    <line x1="50" y1="150" x2="750" y2="150" class="grid"/>
    <line x1="50" y1="250" x2="750" y2="250" class="grid"/>
    <line x1="50" y1="350" x2="750" y2="350" class="grid"/>
    <line x1="50" y1="450" x2="750" y2="450" class="grid"/>
  </g>

  <!-- Axes -->
  <line x1="50" y1="50" x2="50" y2="480" class="axis"/>
  <line x1="50" y1="480" x2="750" y2="480" class="axis"/>

  <!-- Theoretical quadratic convergence line -->
  <polyline class="theory" points="
    100,450 150,400 200,320 250,220 300,120 350,80 400,50
  "/>

  <!-- Method 1: Linear convergence -->
  <polyline class="method-1" points="
    100,440 150,380 200,330 250,280 300,230 350,180 400,130 450,90 500,60
  "/>
  <circle cx="100" cy="440" class="marker"/>
  <circle cx="150" cy="380" class="marker"/>
  <circle cx="250" cy="280" class="marker"/>
  <circle cx="350" cy="180" class="marker"/>
  <circle cx="450" cy="90" class="marker"/>

  <!-- Method 2: Superlinear convergence -->
  <polyline class="method-2" points="
    100,430 150,360 200,280 250,200 300,120 350,70 400,50
  "/>
  <circle cx="100" cy="430" fill="#FF6B6B" r="3"/>
  <circle cx="150" cy="360" fill="#FF6B6B" r="3"/>
  <circle cx="250" cy="200" fill="#FF6B6B" r="3"/>
  <circle cx="350" cy="70" fill="#FF6B6B" r="3"/>

  <!-- Axis labels -->
  <text x="400" y="530" text-anchor="middle" style="font: 14px Arial;">
    Iteration (log scale)
  </text>
  <text x="10" y="265" text-anchor="middle" style="font: 14px Arial;" transform="rotate(-90 10 265)">
    Error (log scale)
  </text>

  <!-- Tick labels (log scale) -->
  <text x="45" y="490" text-anchor="end" style="font: 10px Arial;">10^0</text>
  <text x="45" y="400" text-anchor="end" style="font: 10px Arial;">10^-2</text>
  <text x="45" y="300" text-anchor="end" style="font: 10px Arial;">10^-4</text>
  <text x="45" y="100" text-anchor="end" style="font: 10px Arial;">10^-8</text>
  
  <text x="100" y="510" text-anchor="middle" style="font: 10px Arial;">1</text>
  <text x="250" y="510" text-anchor="middle" style="font: 10px Arial;">10</text>
  <text x="400" y="510" text-anchor="middle" style="font: 10px Arial;">100</text>
  <text x="600" y="510" text-anchor="middle" style="font: 10px Arial;">10^4</text>

  <!-- Legend -->
  <g transform="translate(550, 100)">
    <rect x="0" y="0" width="180" height="120" fill="white" stroke="black" stroke-width="1"/>
    
    <polyline class="method-1" points="10,20 40,20"/>
    <text x="50" y="25" style="font: 12px Arial;">Linear (r=1)</text>
    
    <polyline class="method-2" points="10,50 40,50"/>
    <text x="50" y="55" style="font: 12px Arial;">Superlinear (r≈1.5)</text>
    
    <polyline class="theory" points="10,80 40,80"/>
    <text x="50" y="85" style="font: 12px Arial;">Quadratic (r=2)</text>
  </g>

  <text x="10" y="570" style="font-size: 10px; fill: #666;">
    Convergence rate analysis | Reference: tolerance 1e-8
  </text>
</svg>
```

#### 2.2.6 Error Distribution Plot (Histogram)

**File: `error_distribution.svg`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <title>Error Distribution</title>
  <defs>
    <style>
      .bar { fill: #2196F3; stroke: #1976D2; stroke-width: 1; }
      .axis { stroke: black; stroke-width: 2; }
      .grid { stroke: #e0e0e0; stroke-width: 1; }
    </style>
  </defs>

  <text x="400" y="30" style="font-size: 20px; font-weight: bold; text-anchor: center;">
    Error Distribution Across Grid Points
  </text>

  <!-- Grid and axes -->
  <line x1="80" y1="50" x2="80" y2="450" class="axis"/>
  <line x1="80" y1="450" x2="750" y2="450" class="axis"/>

  <!-- Histogram bars -->
  <rect x="100" y="380" width="25" height="70" class="bar"/>
  <rect x="130" y="250" width="25" height="200" class="bar"/>
  <rect x="160" y="120" width="25" height="330" class="bar"/>
  <rect x="190" y="80" width="25" height="370" class="bar"/>
  <rect x="220" y="100" width="25" height="350" class="bar"/>
  <rect x="250" y="260" width="25" height="190" class="bar"/>
  <rect x="280" y="380" width="25" height="70" class="bar"/>

  <!-- Normal distribution overlay -->
  <polyline fill="none" stroke="#FF6B6B" stroke-width="2" points="
    85,420 100,410 120,360 140,280 160,180 180,100 200,60 220,50 240,70 260,140 280,240 300,330 320,400 330,420
  " stroke-dasharray="5,5"/>

  <!-- Axis labels -->
  <text x="400" y="500" text-anchor="middle" style="font: 14px Arial;">Error Magnitude (|u_computed - u_exact|)</text>
  <text x="20" y="250" text-anchor="middle" style="font: 14px Arial;" transform="rotate(-90 20 250)">Frequency</text>

  <!-- X-axis tick labels -->
  <text x="100" y="475" text-anchor="middle" style="font: 11px Arial;">0.0</text>
  <text x="190" y="475" text-anchor="middle" style="font: 11px Arial;">1e-4</text>
  <text x="280" y="475" text-anchor="middle" style="font: 11px Arial;">2e-4</text>
  <text x="370" y="475" text-anchor="middle" style="font: 11px Arial;">3e-4</text>

  <!-- Statistics box -->
  <g transform="translate(500, 100)">
    <rect x="0" y="0" width="220" height="180" fill="white" stroke="black" stroke-width="1"/>
    <text x="10" y="25" style="font: 12px Arial; font-weight: bold;">Statistics</text>
    <text x="10" y="50" style="font: 11px Arial;">Mean error: 1.23e-4</text>
    <text x="10" y="70" style="font: 11px Arial;">Std dev: 4.56e-5</text>
    <text x="10" y="90" style="font: 11px Arial;">Min error: 1.2e-5</text>
    <text x="10" y="110" style="font: 11px Arial;">Max error: 3.8e-4</text>
    <text x="10" y="130" style="font: 11px Arial;">RMS error: 1.34e-4</text>
    <text x="10" y="150" style="font: 11px Arial;">N_points: 1024</text>
    <text x="10" y="170" style="font: 11px Arial;">Skewness: -0.12</text>
  </g>

  <text x="10" y="570" style="font-size: 10px; fill: #666;">
    Red curve: Gaussian fit | Domain: 64x64 structured grid
  </text>
</svg>
```

### 2.3 SVG Export Function Template (Pseudo-Goth)

```goth
fn svg_header(width: Int, height: Int, title: String) -> String {
  format_string(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
    "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    "  <title>{}</title>\n",
    width, height, title
  )
}

fn svg_line(x1: Float, y1: Float, x2: Float, y2: Float, 
            stroke: String, width: Float) -> String {
  format_string(
    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
    x1, y1, x2, y2, stroke, width
  )
}

fn svg_polyline(points: Array[(Float, Float)], stroke: String, 
                width: Float, fill: String) -> String {
  points_str = join(map(points, fn(p) -> String {
    format_string("{},{}", p.0, p.1)
  }), " ")
  format_string(
    "  <polyline points=\"{}\" stroke=\"{}\" stroke-width=\"{}\" fill=\"{}\"/>\n",
    points_str, stroke, width, fill
  )
}

fn svg_circle(cx: Float, cy: Float, r: Float, fill: String) -> String {
  format_string(
    "  <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\"/>\n",
    cx, cy, r, fill
  )
}

fn svg_footer() -> String {
  "</svg>\n"
}

fn generate_convergence_svg(iterations: Array[Int], 
                            errors: Array[Float],
                            filename: String) -> IO[Unit] {
  handle = create_file(filename)
  write(handle, svg_header(800, 600, "Convergence Plot"))
  
  // Scale data to fit SVG coordinates
  max_iter = max(iterations)
  max_error = max(errors)
  scale_x = 700.0 / float(max_iter)
  scale_y = 450.0 / log10(max_error)
  
  points = map(zip(iterations, errors), fn(p) -> (Float, Float) {
    (50.0 + float(p.0) * scale_x, 500.0 - log10(p.1) * scale_y)
  })
  
  write(handle, svg_polyline(points, "#2196F3", 2.0, "none"))
  write(handle, svg_footer())
  close_file(handle)
}
```

---

## Part 3: Integration Strategy

### 3.1 Automatic Output Generation

Each example should include a function that:
1. Generates intermediate results
2. Collects data into arrays
3. Exports to CSV automatically after computation
4. Generates SVG visualization from CSV data

### 3.2 Example Output Structure

For a single example, expect these files:

```
example_newton_raphson/
├── newton_raphson.goth          # Main Goth source
├── output/
│   ├── convergence_data.csv     # Raw numerical data
│   ├── convergence_plot.svg     # Visualization
│   └── analysis.txt             # Convergence metrics
```

### 3.3 Standardized Metadata in Outputs

Both CSV headers and SVG metadata should include:
- Algorithm/method name
- Date and time generated
- Version of Goth compiler
- Parameters used
- Domain/problem description
- Convergence criteria
- Execution time

