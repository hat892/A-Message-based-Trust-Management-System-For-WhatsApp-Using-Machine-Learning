package trust_system_lib;

import core_lib.*;
import java.io.*;
import java.util.*;

/**
 * WhatsTrustTM.java
 * ------------------
 * Extends the WhatsTrust (Socialtrust) algorithm with an independent
 * URL-based trust layer (URLTrust) that acts as a second gate.
 *
 * Final decision requires BOTH to pass:
 *   Gate 1: WhatsTrust user trust >= 0.5  (sender is trustworthy)
 *   Gate 2: URLTrust.isSafe() == true     (URL is not phishing)
 *
 * Decision table:
 *   User trusted + URL safe     → ACCEPTED
 *   User trusted + URL phishing → REJECTED (key contribution!)
 *   User not trusted + any URL  → REJECTED
 *
 * When phishing is detected, a fake bad transaction is injected
 * into WhatsTrust's Subjective Logic to lower the sender's trust
 * score for future transactions.
 */
public class WhatsTrustTM extends Socialtrust {

    // ======================== PATHS ========================

    private static final String PYTHON_PATH =
        "C:\\Users\\Hatna\\PycharmProjects\\PythonProject1\\.venv\\Scripts\\python.exe";

    private static final String BATCH_SCRIPT =
        "C:\\Users\\Hatna\\OneDrive\\Desktop\\selected project\\JavaApplication22\\JavaApplication22\\src\\classify_urls_batch.py";

    private static final String GOOD_MSG_PATH =
        "C:\\Users\\Hatna\\OneDrive\\Desktop\\selected project\\JavaApplication22\\JavaApplication22\\src\\good_messages.txt";

    private static final String BAD_MSG_PATH =
        "C:\\Users\\Hatna\\OneDrive\\Desktop\\selected project\\JavaApplication22\\JavaApplication22\\src\\bad_messages.txt";

    private static final String FWD_MSG_PATH =
        "C:\\Users\\Hatna\\OneDrive\\Desktop\\selected project\\JavaApplication22\\JavaApplication22\\src\\forwarded_msgs.txt";

    private static final String URL_INPUT_FILE =
        "C:\\Users\\Hatna\\OneDrive\\Desktop\\selected project\\JavaApplication22\\JavaApplication22\\src\\urls_to_classify.txt";

    private static final String URL_OUTPUT_FILE =
        "C:\\Users\\Hatna\\OneDrive\\Desktop\\selected project\\JavaApplication22\\JavaApplication22\\src\\url_results.txt";

    // ======================== CONSTANTS ========================

    private static final double PROB_FORWARD_PHISHING = 0.10;
    private static final double PROB_LEGIT_URL        = 0.20;
    private static final double TRUST_THRESHOLD       = 0.5;

    // ======================== FIELDS ========================

    private Network nw;
    private URLTrust urlTrust = new URLTrust();

    private List<String> goodMessages      = new ArrayList<>();
    private List<String> badMessages       = new ArrayList<>();
    private List<String> forwardedMessages = new ArrayList<>();
    private Random random = new Random();

    private Map<String, Integer> urlCache = new HashMap<>();

    // Stats
    private int urlsDetected         = 0;
    private int urlsPhishing         = 0;
    private int forwardedUrlsBlocked = 0;
    private int trustDecreased       = 0;
    private int rejectedByURL        = 0;

    // ======================== CONSTRUCTOR ========================

    public WhatsTrustTM(Network nw) {
        super(nw);
        this.nw = nw;

        goodMessages      = loadFile(GOOD_MSG_PATH);
        badMessages       = loadFile(BAD_MSG_PATH);
        forwardedMessages = loadFile(FWD_MSG_PATH);

        System.out.printf("[WhatsTrust] Loaded %d good, %d bad, %d forwarded messages.\n",
            goodMessages.size(), badMessages.size(), forwardedMessages.size());

        if (goodMessages.isEmpty())      goodMessages.add("Hey are you free? https://www.google.com");
        if (badMessages.isEmpty())       badMessages.add("Win now! http://fake-prize.ru/claim");
        if (forwardedMessages.isEmpty()) forwardedMessages.add("Check this http://phish-site.ru/login");

        System.out.println("[WhatsTrust] Batch classifying all URLs (one Python call)...");
        long t1 = System.currentTimeMillis();
        batchClassifyAllURLs();
        long t2 = System.currentTimeMillis();

        urlTrust.loadCache(urlCache);

        System.out.printf("[WhatsTrust] URL cache ready: %d URLs in %.1f seconds.\n",
            urlCache.size(), (t2 - t1) / 1000.0);
    }

    // ======================== TRUSTALG INTERFACE ========================

    @Override
    public String fileExtension() { return "whatstrust"; }

    @Override
    public String algName() { return "WhatsTrust + URL Classification"; }

    @Override
    public void update(Transaction trans) {
        int sender   = trans.getSend();
        int receiver = trans.getRecv();

        // Step 1: Original WhatsTrust update
        super.update(trans);

        // Step 2: Get message and extract URL
        String message = getMessage(sender);
        String url     = extractURL(message);

        // No URL → WhatsTrust decision stands, nothing to print
        if (url == null) return;

        urlsDetected++;

        // Step 3: Evaluate both gates
        boolean urlSafe     = urlTrust.isSafe(url);
        double  userTrust   = nw.getUserRelation(receiver, sender).getTrust();
        boolean userTrusted = (userTrust >= TRUST_THRESHOLD);

        // ── Print URL transactions only ──
        System.out.println("\n[WhatsTrust] URL Transaction");
        System.out.printf("  Sender %d -> Receiver %d%n", sender, receiver);
        System.out.printf("  URL        : %s%n", url);
        System.out.printf("  User trust : %.4f  |  Gate 1: %s%n",
            userTrust, userTrusted ? "PASS" : "FAIL");
        System.out.printf("  URL check  : Gate 2: %s%n",
            urlSafe ? "PASS (legitimate)" : "FAIL (phishing)");

        if (urlSafe) {
            System.out.printf("  Decision   : %s%n",
                userTrusted ? "ACCEPTED" : "REJECTED - user not trusted");
            return;
        }

        // URL is phishing — Gate 2 fails
        urlsPhishing++;
        if (nw.getUser(sender).isgood()) forwardedUrlsBlocked++;
        if (userTrusted) rejectedByURL++;

        System.out.printf("  Decision   : REJECTED - %s%n",
            userTrusted
                ? "phishing URL detected from trusted sender!"
                : "both gates failed");

        // Apply trust penalty if sender has positive trust
        double trustBefore = nw.getUserRelation(receiver, sender).getTrust();
        if (trustBefore > 0) {
            Transaction fakeBad = new Transaction(
                trans.getCommit(), sender, receiver, trans.getFile(), false
            );
            super.update(fakeBad);
            super.computeTrust(receiver, trans.getCommit());

            double trustAfter = nw.getUserRelation(receiver, sender).getTrust();
            System.out.printf("  Trust      : %.4f -> %.4f  (change: %.4f)%n",
                trustBefore, trustAfter, trustAfter - trustBefore);

            if (trustAfter < trustBefore) trustDecreased++;
        } else {
            System.out.println("  Trust      : already distrusted, no penalty needed");
        }
    }

    @Override
    public void computeTrust(int user, int cycle) {
        super.computeTrust(user, cycle);
    }

    // ======================== MESSAGE LOGIC ========================

    private String getMessage(int userId) {
        if (nw.getUser(userId).isgood()) {
            double r = random.nextDouble();
            if (r < PROB_FORWARD_PHISHING)
                return forwardedMessages.get(random.nextInt(forwardedMessages.size()));
            else if (r < PROB_FORWARD_PHISHING + PROB_LEGIT_URL)
                return goodMessages.get(random.nextInt(goodMessages.size()));
            else
                return "Hey, are you free tonight?";
        } else {
            return badMessages.get(random.nextInt(badMessages.size()));
        }
    }

    private String extractURL(String message) {
        if (message == null) return null;
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(
            "https?://[\\w\\-._~:/?#\\[\\]@!$&'()*+,;=%]+"
        );
        java.util.regex.Matcher m = p.matcher(message);
        return m.find() ? m.group() : null;
    }

    // ======================== BATCH URL CLASSIFICATION ========================

    private void batchClassifyAllURLs() {
        Set<String> allURLs = new LinkedHashSet<>();
        for (String msg : goodMessages)      { String u = extractURL(msg); if (u != null) allURLs.add(u); }
        for (String msg : badMessages)       { String u = extractURL(msg); if (u != null) allURLs.add(u); }
        for (String msg : forwardedMessages) { String u = extractURL(msg); if (u != null) allURLs.add(u); }

        if (allURLs.isEmpty()) {
            System.out.println("[WhatsTrust] No URLs found in messages.");
            return;
        }

        System.out.printf("[WhatsTrust] Found %d unique URLs to classify.\n", allURLs.size());
        List<String> urlList = new ArrayList<>(allURLs);

        try (PrintWriter pw = new PrintWriter(new FileWriter(URL_INPUT_FILE))) {
            for (String url : urlList) pw.println(url);
        } catch (IOException e) {
            System.err.println("[WhatsTrust] Error writing URL input file: " + e.getMessage());
            return;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(
                PYTHON_PATH, BATCH_SCRIPT, URL_INPUT_FILE, URL_OUTPUT_FILE
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line = reader.readLine()) != null)
                System.out.println("[WhatsTrust] " + line);
            process.waitFor();
        } catch (Exception e) {
            System.err.println("[WhatsTrust] Batch classify error: " + e.getMessage());
            return;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(URL_OUTPUT_FILE))) {
            for (String url : urlList) {
                String resultLine = br.readLine();
                if (resultLine != null && resultLine.trim().matches("[01]"))
                    urlCache.put(url, Integer.parseInt(resultLine.trim()));
                else
                    urlCache.put(url, 1);
            }
        } catch (IOException e) {
            System.err.println("[WhatsTrust] Error reading URL results: " + e.getMessage());
        }
    }

    // ======================== FILE LOADER ========================

    private List<String> loadFile(String path) {
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) lines.add(line);
            }
        } catch (IOException e) {
            System.err.println("[WhatsTrust] Could not load: " + path);
        }
        return lines;
    }

    // ======================== STATS ========================

    public void printURLStats() {
        System.out.printf("\n[WhatsTrust] URL Detection Stats:\n");
        System.out.printf("  URLs detected in transactions:          %d\n", urlsDetected);
        System.out.printf("  Phishing URLs detected:                 %d\n", urlsPhishing);
        System.out.printf("  Forwarded phishing from good users:     %d\n", forwardedUrlsBlocked);
        System.out.printf("  Rejected by URL gate (trusted sender):  %d\n", rejectedByURL);
        System.out.printf("  Trust decreased after phishing:         %d\n", trustDecreased);
        System.out.printf("  Phishing detection rate:                %.2f%%\n",
            urlsDetected > 0 ? (urlsPhishing * 100.0 / urlsDetected) : 0.0);
        System.out.printf("\n  [KEY METRIC] Phishing caught from trusted users: %d\n",
            forwardedUrlsBlocked);
        urlTrust.printStats();
    }
}