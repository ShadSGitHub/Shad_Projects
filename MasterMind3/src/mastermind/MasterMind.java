
package mastermind;
//@author Janet Baez

import userinterface.MasterMindUi;
import core.Game;
import javax.swing.JOptionPane;

public class MasterMind 
{

    public static void main(String[] args) 
    {
        System.out.print("Welcome to MasterMind!");
        JOptionPane.showMessageDialog(null,"Let's Play MasterMind!" ); 
        Game game = new Game();   
        
     MasterMindUi ui =  new MasterMindUi(game);
    }
    
}
