
/**
 * @author Janet Baez
 * University of Central Florida
 * Object Oriented
 * MasterMind 3
 */

package userinterface;

import javax.swing.*;
import javax.swing.JPanel;
import javax.swing.BorderFactory;
import java.awt.Dimension;
import core.Codebreaker;
import java.awt.*;
import java.util.*;
import constants.Constants;

public class CodebreakerUi 
{
    //Member variables
    
    JPanel codebreakerAttempt;
    JPanel codebreakerColors;
    Codebreaker codebreaker;
    RoundButton [] buttons;
    RoundButton[][] attempts;
    
    //Getters
    
    public JPanel getCodebreakerAttempt()
    {
        return codebreakerAttempt;
    }
    public JPanel getCodebreakerColors()
    {
        return codebreakerColors;
    }
    public CodebreakerUi(Codebreaker codebreaker)
    {
        initComponents(codebreaker);
    }
    
    //Calling out method initComponents()
    
    private void initComponents(Codebreaker codebreaker)
    {
        this.codebreaker = codebreaker;
        this.codebreakerAttempt = new JPanel();
        this.codebreakerColors = new JPanel();
        
        //Set JPanel Sizes
        
        codebreakerAttempt.setMinimumSize(new Dimension(225, 150));
        codebreakerAttempt.setPreferredSize(new Dimension(225,150));
        codebreakerColors.setMinimumSize(new Dimension(200, 65));
        codebreakerColors.setPreferredSize(new Dimension(200,65));
        
        //Set Border
        
        codebreakerAttempt.setBorder(BorderFactory.createTitledBorder("Codebreaker Attempt"));
        codebreakerColors.setBorder(BorderFactory.createTitledBorder("Codebreaker Colors"));
        
        //Setting layout
        
        codebreakerAttempt.setLayout(new GridLayout(10,4));
        
        //2-D array
        
        attempts = new RoundButton[10][4];
        
        for(int i=0; i < 10; i++)
        {
            for (int j=0; j < 4; j++)
            {
                if(j == 3)
                {
                    attempts[i][j] = new RoundButton();
                    codebreakerAttempt.setEnabled(false);
                }
                else
                {
                    attempts[i][j] = new RoundButton();
                }
                codebreakerAttempt.add(attempts[i][j]);
            }
            
        }
        
        // 1-D array
        
        buttons = new RoundButton[Constants.COLORS];
        
        int index = 0;
         
       for(RoundButton b: buttons)
       {
           RoundButton button = new RoundButton();
           Color colorSetter = Constants.codeColors.get(index);
           button.setBackground(colorSetter);
           button.putClientProperty("color", colorSetter);
           button.setToolTipText(colorSetter + "color");
           codebreakerColors.add(button);
           index++;
       }
    }
}
