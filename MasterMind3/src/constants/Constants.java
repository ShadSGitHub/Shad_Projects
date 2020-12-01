package constants;
//@author Janet Baez

import java.awt.Color;
import java.util.*;

public class Constants 
{
    //Section 1
    public static final ArrayList<Color>codeColors = 
        new ArrayList<Color>(Arrays.asList(
        Color.BLUE, Color.BLACK, Color.ORANGE, Color.WHITE, 
        Color.YELLOW, Color.RED, Color.GREEN, Color.PINK));
    //Section 2
    public static final ArrayList<Color>responseColors =
        new ArrayList<Color>(Arrays.asList(Color.RED, Color.WHITE));
    
// Declering Constants - Sections 3, 4 & 5
    public static final int MAX_ATTEMPTS = 10;
    public static final int MAX_PEGS = 4;
    public static final int COLORS = 8;
}