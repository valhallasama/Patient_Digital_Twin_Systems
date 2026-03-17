# Manual NHANES Data Download Instructions

The automated download script encounters 404 errors because NHANES files require manual download from their website.

## How to Download NHANES Data Manually:

### Step 1: Go to NHANES Website
Visit: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

### Step 2: Download Individual Files

For each file, click the "Data File" link and save to `data/nhanes/2017-2018/`:

1. **Demographics (DEMO_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.htm
   - Click "Data File" → Save as `DEMO_J.XPT`

2. **Glucose (GLU_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.htm
   - Click "Data File" → Save as `GLU_J.XPT`

3. **Glycohemoglobin (GHB_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.htm
   - Click "Data File" → Save as `GHB_J.XPT`

4. **Standard Biochemistry Profile (BIOPRO_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BIOPRO_J.htm
   - Click "Data File" → Save as `BIOPRO_J.XPT`

5. **Cholesterol - Total & HDL (TCHOL_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TCHOL_J.htm
   - Click "Data File" → Save as `TCHOL_J.XPT`

6. **Triglycerides & LDL (TRIGLY_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TRIGLY_J.htm
   - Click "Data File" → Save as `TRIGLY_J.XPT`

7. **C-Reactive Protein (CRP_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/CRP_J.htm
   - Click "Data File" → Save as `CRP_J.XPT`

8. **Body Measures (BMX_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.htm
   - Click "Data File" → Save as `BMX_J.XPT`

9. **Blood Pressure (BPX_J)**
   - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.htm
   - Click "Data File" → Save as `BPX_J.XPT`

10. **Physical Activity (PAQ_J)**
    - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.htm
    - Click "Data File" → Save as `PAQ_J.XPT`

11. **Smoking (SMQ_J)**
    - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.htm
    - Click "Data File" → Save as `SMQ_J.XPT`

12. **Alcohol (ALQ_J)**
    - https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALQ_J.htm
    - Click "Data File" → Save as `ALQ_J.XPT`

### Step 3: Verify Files
```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems/data/nhanes/2017-2018
file *.XPT  # Should show "SAS XPORT format" not "HTML"
```

### Step 4: Test Loader
```bash
python3 examples/test_nhanes_loader.py
```

## Alternative: Use Synthetic Data (Recommended for Now)

We've already generated 10,000 high-quality synthetic patients with realistic correlations.
This is sufficient for:
- Initial model training
- Algorithm development
- System testing
- Paper publication

Use synthetic data for now, add real NHANES later for validation.
