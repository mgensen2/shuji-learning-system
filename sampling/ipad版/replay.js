window.addEventListener('load', () => {
    const canvas = document.getElementById('replayCanvas');
    const ctx = canvas.getContext('2d');
    const fileInput = document.getElementById('csvFileInput');
    const drawBtn = document.getElementById('drawBtn');
    const saveBtn = document.getElementById('saveBtn');

    // --- 設定項目 ---
    const GRID_SIZE = 8;
    const COORD_LIMIT = 200;
    
    // キャンバスサイズ
    let canvasSize = 800; 
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // 現在描画中のデータを保持する変数（再描画用）
    let currentData = [];

    // ----------------------------------------
    // イベントリスナー
    // ----------------------------------------
    drawBtn.addEventListener('click', handleFileSelect);

    if (saveBtn) {
        saveBtn.addEventListener('click', saveImage);
    }

    // ----------------------------------------
    // メイン処理
    // ----------------------------------------
    function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) {
            alert("CSVファイルを選択してください。");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            
            // CSVの種類を判定してパース
            let data = [];
            if (text.includes("event_type")) {
                console.log("Detected Full Data CSV");
                data = parseFullCSV(text);
            } else if (text.includes("Command")) {
                console.log("Detected Plotter CSV");
                data = parsePlotterCSV(text);
            } else {
                alert("不明なCSV形式です。");
                return;
            }

            if (data.length > 0) {
                currentData = data; // データを保存しておく
                // 通常描画：グリッドあり(true), 番号あり(true)
                drawReplay(data, true, true); 
            } else {
                alert("有効なデータが見つかりませんでした。");
            }
        };
        reader.readAsText(file);
    }

    // ----------------------------------------
    // 画像保存処理
    // ----------------------------------------
    function saveImage() {
        // 1. グリッドなし(false)、番号なし(false)で再描画
        if (currentData.length > 0) {
            drawReplay(currentData, false, false); 
        } else {
            // データ未読み込み時（空白の紙として保存する場合）
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            // グリッド等を描かないならこれだけでOK
        }

        // 2. 画像として保存
        const link = document.createElement('a');
        link.download = 'unpitsu_replay.png';
        link.href = canvas.toDataURL('image/png');
        link.click();

        // 3. 画面表示用に元に戻す：グリッドあり(true), 番号あり(true)
        if (currentData.length > 0) {
            drawReplay(currentData, true, true);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGridAndNumbers(true, true);
        }
    }

    // ----------------------------------------
    // CSVパース処理 (Full Data)
    // ----------------------------------------
    function parseFullCSV(text) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim());
        
        const idxType = headers.indexOf('event_type');
        const idxX = headers.indexOf('x');
        const idxY = headers.indexOf('y');
        const idxPress = headers.indexOf('pressure');
        const idxStroke = headers.indexOf('stroke_id');

        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            if (row.length < headers.length) continue;

            data.push({
                type: 'full',
                event_type: row[idxType].trim(),
                x: parseFloat(row[idxX]),
                y: parseFloat(row[idxY]),
                pressure: idxPress !== -1 ? parseFloat(row[idxPress]) : 1.0,
                stroke_id: idxStroke !== -1 ? parseInt(row[idxStroke]) : 0,
                is_drawing: row[idxType].trim() !== 'up'
            });
        }
        return data;
    }

    // ----------------------------------------
    // CSVパース処理 (Plotter Data)
    // ----------------------------------------
    function parsePlotterCSV(text) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim());
        
        const idxCmd = headers.indexOf('Command');
        const idxX = headers.indexOf('X');
        const idxY = headers.indexOf('Y');
        const idxZ = headers.indexOf('Z');
        
        if (idxCmd === -1 || idxX === -1 || idxY === -1) {
            alert("プロッタCSVフォーマットエラー: 必要な列が見つかりません。");
            return [];
        }

        const data = [];
        let strokeCount = 0;

        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            if (row.length < headers.length) continue;

            const cmd = row[idxCmd].trim();
            const x = parseFloat(row[idxX]);
            const y = parseFloat(row[idxY]);
            const z = idxZ !== -1 ? parseFloat(row[idxZ]) : 0;

            let isDrawing = false;
            
            if (cmd === 'A0' || cmd === 'G0') {
                strokeCount++; 
                isDrawing = false; 
            } else if (cmd === 'A1' || cmd === 'G1') {
                isDrawing = true;
            } else if (cmd === 'D1') {
                isDrawing = true; 
            }

            data.push({
                type: 'plotter',
                command: cmd,
                x: x,
                y: y,
                pressure: z,
                stroke_id: strokeCount,
                is_drawing: isDrawing
            });
        }
        return data;
    }

    // ----------------------------------------
    // 描画処理
    // 引数 drawGrid: trueなら罫線あり
    // 引数 drawNumbers: trueなら番号あり
    // ----------------------------------------
    function drawReplay(data, drawGrid = true, drawNumbers = true) {
        // キャンバスをクリア
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // ★重要: 保存時に背景が透明にならないよう白で塗りつぶす
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // 背景・グリッド描画
        drawGridAndNumbers(drawGrid, drawNumbers);

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const colors = ['#000000', '#FF0000', '#0000FF', '#008000', '#FFA500', '#800080'];

        const toPixel = (val, isX) => {
            if (isX) return ((val / COORD_LIMIT) + 1.0) * canvasSize;
            else return (val / -COORD_LIMIT) * canvasSize;
        };

        let isPenDown = false;
        let lastX = null;
        let lastY = null;

        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            const px = toPixel(point.x, true);
            const py = toPixel(point.y, false);
            
            const pressure = point.pressure || 0;
            const lineWidth = Math.max(1, pressure * 1.5);

            const colorIdx = (point.stroke_id - 1) % colors.length;
            const color = colors[colorIdx >= 0 ? colorIdx : 0];

            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = color;
            ctx.fillStyle = color;

            if (point.type === 'full') {
                if (point.event_type === 'down') {
                    isPenDown = true;
                    lastX = px; lastY = py;
                    ctx.beginPath();
                    ctx.arc(px, py, lineWidth / 2, 0, Math.PI * 2);
                    ctx.fill();
                } else if (point.event_type === 'move') {
                    if (isPenDown && lastX !== null) {
                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(px, py);
                        ctx.stroke();
                        lastX = px; lastY = py;
                    }
                } else if (point.event_type === 'up') {
                    isPenDown = false;
                    lastX = null; lastY = null;
                }
            } else {
                const isMoveCommand = (point.command === 'A0' || point.command === 'G0');
                const isDrawCommand = (point.command === 'A1' || point.command === 'G1' || point.command === 'D1');

                if (isMoveCommand) {
                    isPenDown = false;
                    lastX = px; lastY = py;
                } else if (isDrawCommand) {
                    if (lastX !== null) {
                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(px, py);
                        ctx.stroke();
                        
                        if (point.command === 'D1') {
                            ctx.beginPath();
                            ctx.arc(px, py, lineWidth * 1.5, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    } else {
                        ctx.beginPath();
                        ctx.arc(px, py, lineWidth / 2, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    lastX = px; lastY = py;
                }
            }
        }
    }

    // ----------------------------------------
    // グリッドと番号の描画
    // ----------------------------------------
    function drawGridAndNumbers(drawGrid = true, drawNumbers = true) {
        const cellSize = canvasSize / GRID_SIZE;

        // グリッド線と外枠 (drawGridがtrueの時だけ描画)
        if (drawGrid) {
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 1; i < GRID_SIZE; i++) {
                ctx.moveTo(i * cellSize, 0);
                ctx.lineTo(i * cellSize, canvas.height);
                ctx.moveTo(0, i * cellSize);
                ctx.lineTo(canvas.width, i * cellSize);
            }
            ctx.stroke();

            // 外枠
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.strokeRect(0, 0, canvas.width, canvas.height);
        }

        // セル番号 (drawNumbersがtrueの時だけ描画)
        if (drawNumbers) {
            ctx.fillStyle = '#ccc'; 
            ctx.font = '20px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            for (let y = 0; y < GRID_SIZE; y++) {
                for (let x = 0; x < GRID_SIZE; x++) {
                    const cellId = (y * GRID_SIZE) + x + 1; 
                    const centerX = x * cellSize + (cellSize / 2);
                    const centerY = y * cellSize + (cellSize / 2);
                    ctx.fillText(cellId, centerX, centerY);
                }
            }
        }
    }
    
    // 初期表示（両方あり）
    drawGridAndNumbers(true, true);
});