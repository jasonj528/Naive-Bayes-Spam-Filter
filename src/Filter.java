/* Jason Johnson
 * CSC 425
 * Program 3
 * 4/7/17
 */

/* This program is a spam filter employing Naive Bayes Classification. The system 
 * is trained using a training set of emails already identified as spam or "ham". 
 * A set of test messages can then be classified as spam or not with a naive 
 * implementation of Bayes theorem that calculates the ratio of the likelihood 
 * it is spam over that of it being ham. After any number of test cases are 
 * classified, the token probabilities can be updated to include the test cases 
 * as a part of the testing set.
 */

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;

import javax.mail.BodyPart;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;

import org.apache.commons.io.IOUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document.OutputSettings;
import org.jsoup.safety.Whitelist;

public class Filter {
    // file containing training filenames and whether they are spam or not
    static final String LABELS = "SPAMTrain.label";
    static final String TRAINING = "TRAINING/"; // path to the training set
    static final String TESTING = "TESTING/TEST_";  // path to the testing set
    static final int TRAIN_SAMPLE = 2000;   // # emails to train with
    static final int TEST_SAMPLE = 5;       // # emails to test with at a time
    // used to trim the tokens down to relevant ones
    static final double SIG_THRESHOLD = .0001;
    
    static BufferedReader labels;   // reader for the labels file
    static boolean init = false;    // whether tokens have been generated or not
    static int testOffset = 0;      // test cases already covered
    static int spam;        // number of spam emails
    static int ham;         // number of non-spam emails
    static double spamPrior;   // calculated from occurence of spam in dataset
    // mapping of tokens to a pair consisting of the conditional probabilities 
    // of the token appearing given spam/ham
    static HashMap<String, Pair<Double, Double>> prob = new HashMap<>();
    // mapping of prior test cases to their results
    static HashMap<Set<String>,Boolean> tested = new HashMap<>();
    
    public static void main(String[] args) {
        int choice;
        Scanner scn = new Scanner(System.in);
        try {
            labels = new BufferedReader(new FileReader(LABELS));
            labels.mark(55000); // mark beginning of file to reset to
        } catch (IOException e) {
            System.out.println("Error in setting up label file: " + e);
            e.printStackTrace();
        }
        
        if (!init)  { // token sets have not been created
            System.out.println("Initializing Tokens...");
            updateTokens();
            init = true;
        }
        do {
            System.out.format("1. Update Tokens%n2. Print Tokens%n"
                    + "3. Test Mail%n4. Set Next Test Index%n0. Quit%n");
            try {
                choice = Integer.parseInt(scn.nextLine().split("\\s")[0]);
            } 
            catch (NumberFormatException e) {
                choice = -1;
            }
            switch (choice) {
            case 1: updateTokens(); break;
            case 2: printTokens(); break;
            case 3: testMail(); break;
            case 4: try {
                System.out.println("Enter the number for the next test case:");
                int n = Integer.parseInt(scn.nextLine().split("\\s")[0]);
                testOffset = n > 0 ? n : testOffset;
            } 
            catch (NumberFormatException e) {
            }
            break;
            case 0: System.out.println("Quitting."); break;
            default: System.out.println("Invalid Input."); break;
            }
        } while (choice != 0);
        scn.close();
        try {
            labels.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /* Given the set of tokens appearing an an email, this function 
     * will decide whether or not to classify it as Spam employing 
     * naive Bayes classification. Classification as spam is the 
     * evaluation of the formula log(P(S|E)/P(H|E). A positive value 
     * for this means a value > 1 for P(S|E)/P(H|E), which corresponds 
     * to the value of P(S|E) > P(H|E) meaning the email is more likely 
     * to be spam than ham (non-spam).
     */
    
    private static boolean isSpam(Set<String> set) {
        // log P(S) / P(H)
        double ratio = Math.log(spamPrior / (1 - spamPrior));
        // sum log P(E|S) / P(E|H)
        for (String str : set) {
            Pair<Double, Double> pair = prob.get(str);
            ratio += Math.log(pair.left() / pair.right());
        }
        System.out.format("[log base e of the ratio likelihood of "
                + "spam/ham for following message: %8.4f%n]", ratio);
        return ratio > 0;
    }
    
    /* Updates the total number of tokens and populates the HashMaps key-value 
     * pairs corresponding to the probability of significant tokens based on 
     * the classification of an email. Whether or not a token is considered 
     * significant depends on how close to .5 its probability of occurring 
     * in a spam email is.
     */
    private static void updateTokens() {
        try {
            labels.reset();
            prob = new HashMap<String, Pair<Double, Double>>();
            boolean isSpam;
            
            spam = ham = 0;
            
            for (int i = 0; i < TRAIN_SAMPLE; i++) {
                // spl[0] is whether the message is spam, spl[1] is the filename
                String[] spl = labels.readLine().split("\\s");
                isSpam = Integer.parseInt(spl[0]) == 0;
                
                // update spam/ham totals
                if (isSpam)
                    spam++;
                else
                    ham++;
                
                // Construct the email message from the .eml file
                Properties prop = System.getProperties();
                prop.put("mail.host", "smtp.example.com");
                prop.put("mail.transport.protocol", "smtp");
                // following two properties are to get past a few parsing errors
                prop.put("mail.mime.multipart.ignoreexistingboundaryparameter", true);
                prop.put("mail.mime.parameters.strict", false);
                Session sess = Session.getDefaultInstance(prop, null);
                MimeMessage message = new MimeMessage(sess, 
                        new FileInputStream(TRAINING + spl[1]));
                
                // Split the content of a message, delimited by spaces and punctuation, and 
                // construct a HashSet to avoid duplicate entries, since we only care about 
                // whether a token appears or not (at least for now).
                String strMessage = String.format("%s%n%s", message.getSubject(), 
                        getBody(message)).toLowerCase();
                for (String str : new HashSet<String>(Arrays.asList(
                        strMessage.split("[\\p{Punct}\\s]")))) {
                    if (!prob.containsKey(str))
                        prob.put(str, isSpam ? new Pair<Double, Double>(1.0,0.0) 
                                : new Pair<Double, Double>(0.0,1.0));
                    else {
                        Pair<Double, Double> pair = prob.get(str);
                        if (isSpam)
                            pair.setLeft(pair.left() + 1);
                        else 
                            pair.setRight(pair.right() + 1);
                    }
                }
            }
            
            // loop over prior test sets if any
            for (Set<String> set : tested.keySet()) {
                isSpam = tested.get(set);
                if (isSpam)
                    spam++;
                else
                    ham++;
                for (String str : set) {
                    if (!prob.containsKey(str))
                        prob.put(str,  isSpam ? new Pair<Double, Double>(1.0,0.0)
                                : new Pair<Double, Double>(0.0,1.0));
                    else {
                        Pair <Double, Double> pair = prob.get(str);
                        if (isSpam)
                            pair.setLeft(pair.left() + 1);
                        else 
                            pair.setRight(pair.right() + 1);
                    }
                }
            }
            
            // Determine probabilities by determining the percentage
            // of emails in a class which contain the token. 
            ArrayList<String> removeKeys = new ArrayList<>();
            for (String key : prob.keySet()) {
                Pair<Double, Double> pair = prob.get(key);
                pair.setLeft((pair.left() + 1) / (spam + 2));
                pair.setRight((pair.right() + 1) / (ham + 2));
                if (Math.abs(pair.left() - pair.right()) < SIG_THRESHOLD) 
                    removeKeys.add(key);
            }
            
            for (String key : removeKeys) prob.remove(key);
            
            spamPrior = ((double) spam) / (spam + ham);
            System.out.println("Tokens updated.");
        } catch (MessagingException e) {
            System.out.println("Error in parsing email: " + e);
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("Error reading from file: " + e);
            e.printStackTrace();
        }
        
    }
    
    /* Prints out each token and its conditional probabilities */
    private static void printTokens() {
        System.out.println("Printing tokens and conditional probabilities:");
        System.out.format("%-15.15s   %-15.15s %-15.15s%n", "Tokens", "Spam", "Not-Spam");
        for (String str : prob.keySet()) {
            System.out.format("%-15.15s : %.14f %.14f%n", 
                    str, prob.get(str).left(), prob.get(str).right());
        }
    }
    
    /* Classifies mail in a loop, TEST_SAMPLE emails at a time. 
     * Test cases are added to a map to be used for further testing when 
     * tokens are updated.
     */
    private static void testMail() { 
            try {
                for (int i = 0; i < TEST_SAMPLE; i++) {
                    // Construct email data from .eml file
                    Properties prop = System.getProperties();
                    prop.put("mail.host", "smtp.example.com");
                    prop.put("mail.transport.protocol", "smtp");
                    // following two properties are to get past a few parsing errors
                    prop.put("mail.mime.multipart.ignoreexistingboundaryparameter", true);
                    prop.put("mail.mime.parameters.strict", false);
                    Session sess = Session.getDefaultInstance(prop, null);
                    String fn = TESTING + String.format("%05d.eml", (testOffset + i));
                    MimeMessage message = new MimeMessage(sess, new FileInputStream(fn));
                    
                    // create a set with the tokens present in the given email
                    String body = getBody(message);
                    String strMessage = String.format("%s%n%s", message.getSubject(),
                            body).toLowerCase();
                    Set<String> set = new HashSet<String>(Arrays.asList(
                            strMessage.split("[\\p{Punct}\\s]")));
                    

                    Set<String> words = new HashSet<>(set);
                    
                    // remove tokens that aren't in the keyset of prob
                    ArrayList<String> removeKeys = new ArrayList<>();
                    for (String str : set) {
                        if (!prob.containsKey(str))
                            removeKeys.add(str);
                    }
                    
                    for (String str : removeKeys)
                        set.remove(str);
                    
                    boolean isSpam = isSpam(set);
                    
                    // add to tested cases
                    tested.put(words, isSpam);
                    
                    // Output the subject, address, and body of a message
                    // and whether it was classified as spam or not
                    System.out.format("Subject: %s%nFrom: %s%nBody: "
                            + "%s%n%nDetected to be: %s%n%n%n", 
                            message.getSubject(), message.getFrom()[0], 
                            body, isSpam ? "Spam" : "Not Spam");
                }
                
                // add the sample size to the offset
                testOffset += TEST_SAMPLE;
                
            } catch (MessagingException e) {
                System.out.println("Error in parsing email: " + e);
                e.printStackTrace();
            } catch (IOException e) {
                System.out.println("Error reading from file: " + e);
                e.printStackTrace();
            }
    }
    
    /* Attempts to extract a string from an email Message */
    private static String getBody(Message message) 
            throws MessagingException, IOException {
        String res = "";
        if (message.isMimeType("text/plain")) 
            return message.getContent().toString();
        else if (message.isMimeType("multipart/*")) {
            return getTextFromMMP((MimeMultipart) message.getContent());
        }
        else if (message.isMimeType("text/html")) {
            // html mime types tend to return a byte array stream
            String html;
            if (message.getContent() instanceof InputStream) {
                byte[] b = IOUtils.toByteArray(
                        (InputStream) message.getContent());
                html = new String(b);
            }
            else
                html = (String) message.getContent();
            String text = String.format("%s%n", Jsoup.parse(html).text());
            return text;
        }
        return res;
    }
    
    /* In the case of a message resulting in a MimeMultipart object, this
     * function attempts to extract the text recursively. The handling of 
     * each type is about the same as in getBody().
     */
    private static String getTextFromMMP(MimeMultipart mmp) 
            throws MessagingException, IOException {
        String res = "";
        int cnt = mmp.getCount();
        for (int i = 0; i < cnt; i++) {
            BodyPart bp = mmp.getBodyPart(i);
            if (bp.isMimeType("text/plain")) {
                res += String.format("%s%n", bp.getContent());
                break;
            }
            else if (bp.isMimeType("multipart/*")) {
                res += String.format("%s%n", 
                        getTextFromMMP((MimeMultipart) bp.getContent()));
            }
            else if (bp.isMimeType("text/html")) {
                // html mime types tend to return a byte array stream
                String html;
                if (bp.getContent() instanceof InputStream) {
                    byte[] b = IOUtils.toByteArray(
                            (InputStream) bp.getContent());
                    html = new String(b);
                }
                else
                    html = (String) bp.getContent();
                String text = String.format("%s%n", 
                        Jsoup.clean(html, "", Whitelist.none(), 
                                new OutputSettings().prettyPrint(false)));
                res += text;
            }
        }
        return res;
    }
    
    /* Helper class describing a pair of values. Only really used to 
     * avoid needing two mappings for spam and ham conditional probabilities. 
     * A double[2] might've honestly been more simple. 
     */
    static class Pair<L, R> {
        private L left;
        private R right;
        
        public Pair(L left, R right) {
            this.left = left;
            this.right = right;
        }
        
        public L left() { return left; }
        public R right() { return right; }
        
        public void setLeft(L left) { this.left = left; }
        public void setRight(R right) { this.right = right; }
    }
    
    
}
